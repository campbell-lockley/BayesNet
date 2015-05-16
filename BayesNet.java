/* ****************************************************************************************************************** *
 * Name:	BayesNet.java
 * Description:	Peforms approximate inference calculations on a small pre-defined simple bayesian network.
 * 		
 * 		Based on BayesNet.java from:
 * 			http://elearn.waikato.ac.nz/pluginfile.php/701717/mod_resource/content/8/BayesNet.java
 * 
 * Author:	Campbell Lockley	studentID: 1178618	Based on BayesNet.java from
 * Date:	15/05/15
 * ****************************************************************************************************************** */

/** Simple class for approximate inference based on the Poker-game network. */
public class BayesNet {

	/** Inner class for representing a node in the network. */
	private class Node {
		/* Private members */
		private String name;		// The name of the node
		private Node[] parents;		// The parent nodes
		private Node[] children;	// The child nodes
		private double[] probs;		// The probabilities for the CPT
		public boolean value;		// The current value of the node

		/** Initializes the node. */
		private Node(String n, Node[] pa, double[] pr) {
			name = n;
			parents = pa;
			children = new Node[0];
			probs = pr;
		}

		/**
		 * Returns conditional probability of value "true" for the current node based on the values of the 
		 * parent nodes.
		 * 
		 * @return The conditional probability of this node, given its parents.
		 */
		private double conditionalProbability() {
			int index = 0;
			for (int i = 0; i < parents.length; i++) {
				if (parents[i].value == false) index += Math.pow(2, parents.length - i - 1);
			}
			return probs[index];
		}
	}

	/* The list of nodes in the Bayes net */
	private Node[] nodes;

	/* Random number generator for approximate inference methods */
	private java.util.Random rand;

	/* Indices into a vector of values, N[], for a node, X, where X = { TRUE | FALSE } */
	private static final int TRUE = 0;
	private static final int FALSE = 1;

	/* A collection of examples describing whether Bot B is { cocky, bluffing } */
	public static final boolean[][] BBLUFF_EXAMPLES = { { true,  true  },
			{ true,  true  }, { true,  true  }, { true,  false },
			{ true,  true  }, { false, false }, { false, false },
			{ false, true  }, { false, false }, { false, false },
			{ false, false }, { false, false }, { false, true  } };

	/** Constructor that sets up the Poker-game network. */
	public BayesNet() {
		rand = new java.util.Random(System.currentTimeMillis());
		nodes = new Node[7];
		nodes[0] = new Node("B.Cocky", 
					new Node[] {}, 
					new double[] { 0.05 });
		nodes[1] = new Node("B.Bluff", 
					new Node[] { nodes[0] }, 
					calculateBBluffProbabilities(BBLUFF_EXAMPLES));
		nodes[2] = new Node("A.Deals", 
					new Node[] {}, 
					new double[] { 0.5 });
		nodes[3] = new Node("A.GoodHand", 
					new Node[] { nodes[2] }, 
					new double[] { 0.75, 0.5 });
		nodes[4] = new Node("B.GoodHand", 
					new Node[] { nodes[2] }, 
					new double[] { 0.4, 0.5 });
		nodes[5] = new Node("B.Bets", 
					new Node[] { nodes[1], nodes[4] }, 
					new double[] { 0.95, 0.7, 0.9, 0.01 });
		nodes[6] = new Node("A.Wins", 
					new Node[] { nodes[3], nodes[4] }, 
					new double[] { 0.45, 0.75, 0.25, 0.55 });
		nodes[0].children = new Node[] { nodes[1] };
		nodes[1].children = new Node[] { nodes[5] };
		nodes[2].children = new Node[] { nodes[3], nodes[4] };
		nodes[3].children = new Node[] { nodes[6] };
		nodes[4].children = new Node[] { nodes[5], nodes[6] };
	}

	/** Prints the current state of the network to standard out. */
	public void printState() {
		for (int i = 0; i < nodes.length; i++) System.out.print(nodes[i].value + "\t");
		System.out.println();
	}

	/**
	 * Calculates the probability that Bot B will bluff based on whether it is cocky or not.
	 * 
	 * @param bluffInstances A set of training examples in the form { cocky, bluff } from which to compute the 
	 * 		probabilities.
	 * @return The probability that Bot B will bluff when it is { cocky, !cocky }.
	 */
	public double[] calculateBBluffProbabilities(boolean[][] bluffInstances) {
		double[] probabilities = new double[2];
		final int COCKY = 0, BLUFF = 1;
		int total = bluffInstances.length;
		int numCocky = 0;

		/* Calculate conditional probability table for B.Bluff using trainging data in BBLUFF_EXAMPLES */
		for (int i = 0; i < bluffInstances.length; i++) {
			if (bluffInstances[i][COCKY] == true) numCocky++;			// Find P(Cocky)

			if (bluffInstances[i][BLUFF] == true) {
				if (bluffInstances[i][COCKY] == true) probabilities[0]++;	// Find P(Bluff^Cocky)
				else probabilities[1]++;					// Find P(Bluff^~Cocky)
			}
		}
		probabilities[0] = probabilities[0] / numCocky;					// Calc. P(Bluff|Cocky)
		probabilities[1] = probabilities[1] / (total - numCocky);			// Calc. P(Bluff|~Cocky)

		return probabilities;
	}

	/**
	 * This method calculates the exact probability of a given event occurring, where all variables are assigned 
	 * a given evidence value.
	 *
	 * @param evidenceValues The values of all nodes.
	 * @return -1 if the evidence does not cover every node in the network. Otherwise a probability between 0 
	 * 	and 1.
	 */
	public double calculateExactEventProbability(boolean[] evidenceValues) {
		/* Only performs exact calculation for all evidence known. */
		if (evidenceValues.length != nodes.length) return -1;

		double prob = 1, p;

		/* Update bayesian network with current evidence */
		for (int i = 0; i < evidenceValues.length; i++) nodes[i].value = evidenceValues[i];

		/* P(x1, ..., xn) = P(x1|PARENTS(x1)) * ... * P(xn|PARENTS(xn)) */
		for (int i = 0; i < evidenceValues.length; i++) {
			p = nodes[i].conditionalProbability();
			prob *= (evidenceValues[i] == true) ? p : (1 - p);
		}

		return prob;
	}

	/**
	 * This method assigns new values to the nodes in the network by sampling from the joint distribution (based on 
	 * PRIOR-SAMPLE method from text book/slides).
	 */
	public void priorSample() {
		/* Order of nodes in nodes[] is parents of dependancies first, so priorSample() can simply iterate 
		   through node list to find correct contidional probabilities */
		for (int i = 0; i < nodes.length; i++) {
			nodes[i].value = (rand.nextDouble() < nodes[i].conditionalProbability()) ? true : false;
		}
	}

	/**
	 * Checks if the current model of the bayesian network (i.e. the current values of the nodes in nodes[]) is 
	 * consistent with the specified evidence - i.e. if nodes[eNodes[x]].value == eValues[x] for all x = index in 
	 * nodes[].
	 * 
	 * @param eNodes The indicies of the evidence nodes in nodes[].
	 * @param eValues The value of the evidence nodes, in the order specified by eNodes.
	 * @return True if the current model of the bayesian network is consistent with the evidence, false otherwise.
	 */
	private boolean consistent(int[] eNodes, boolean[] eValues) {
		for (int i = 0; i < eNodes.length; i++) {
			if (nodes[eNodes[i]].value != eValues[i]) return false;
		}

		return true;
	}

	/**
	 * Rejection sampling. Returns probability of query variable being true given the values of the evidence 
	 * variables, estimated based on the given total number of samples (see REJECTION-SAMPLING method from text 
	 * book/slides).
	 * 
	 * The nodes/variables are specified by their indices in the nodes array. The array evidenceValues has one 
	 * value for each index in indicesOfEvidenceNodes. See also examples in main().
	 * 
	 * @param queryNode The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes The indices of the evidence nodes.
	 * @param evidenceValues The values of the indexed evidence nodes.
	 * @param N The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the evidence.
	 */
	public double rejectionSampling(int queryNode, int[] indicesOfEvidenceNodes, boolean[] evidenceValues, 
			int N) {
		assert (queryNode < nodes.length);
		assert (indicesOfEvidenceNodes.length == evidenceValues.length);
		
		int count = 0, numConsistent = 0;

		/* Take N samples, throw away samples which are inconsistent with the evidence, and count number of 
		   samples in which queryNode is true */
		for (int i = 0; i < N; i++) {
			priorSample();
			if (consistent(indicesOfEvidenceNodes, evidenceValues)) {
				numConsistent++;
				if (nodes[queryNode].value == true) count++;
			}
		}

		/* Return the normalised true value */
		return count / (double)numConsistent;
	}

	/**
	 * Checks whether node nodes[nodeIndex] is an evidence node and, if it is, sets the node to the observed value.
	 * 
	 * @param nodeIndex The index in nodes[] of the node to check.
	 * @param eNodes The indicies of the evidence nodes in nodes[].
	 * @param eValues The value of the evidence nodes, in the order specified by eNodes.
	 * @return True if this node is an evidence node, false otherwise.
	 */
	private boolean isEvidence(int nodeIndex, int[] eNodes, boolean[] eValues) {
		for (int i = 0; i < eNodes.length; i++) {
			assert (eNodes[i] < nodes.length);

			if (eNodes[i] == nodeIndex) {
				nodes[nodeIndex].value = eValues[i];
				return true;
			}
		}

		return false;
	}

	/**
	 * This method assigns new values to the non-evidence nodes in the network and computes a weight based on the 
	 * evidence nodes (based on WEIGHTED-SAMPLE method from text book/slides).
	 * 
	 * The evidence is specified as in the case of rejectionSampling().
	 * 
	 * @param indicesOfEvidenceNodes The indices of the evidence nodes.
	 * @param evidenceValues The values of the indexed evidence nodes.
	 * @return The weight of the event occurring.
	 */
	public double weightedSample(int[] indicesOfEvidenceNodes, boolean[] evidenceValues) {
		double w = 1, p;

		/* Sample non-evidence nodes and return a weight based on the evidence nodes */
		for (int i = 0; i < nodes.length; i++) {
			if (isEvidence(i, indicesOfEvidenceNodes, evidenceValues)) {
				/* If nodes[i] is an evidence node compute weight */
				p = nodes[i].conditionalProbability();
				w *= (nodes[i].value == true) ? p : (1 - p);
			} else {
				/* If nodes[i] not an evidence node assign a random sample */
				nodes[i].value = (rand.nextDouble() < nodes[i].conditionalProbability()) ? true : false;
			}
		}

		return w;
	}

	/**
	 * Likelihood weighting. Returns probability of query variable being true given the values of the evidence 
	 * variables, estimated based on the given total number of samples (see LIKELIHOOD-WEIGHTING method from text 
	 * book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes The indices of the evidence nodes.
	 * @param evidenceValues The values of the indexed evidence nodes.
	 * @param N The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the 
	 * 	evidence.
	 */
	public double likelihoodWeighting(int queryNode, int[] indicesOfEvidenceNodes, boolean[] evidenceValues, 
			int N) {
		assert (queryNode < nodes.length);
		assert (indicesOfEvidenceNodes.length == evidenceValues.length);

		double w, weight[] = {0, 0};

		/* Take N weighted samples, keeping a vector of sums of the weights based on the query node being true 
		   or false */
		for (int i = 0; i < N; i++) {
			w = weightedSample(indicesOfEvidenceNodes, evidenceValues);
			weight[(nodes[queryNode].value == true) ? TRUE : FALSE] += w;
		}

		/* Return the normalised true value */
		return weight[TRUE] / (weight[TRUE] + weight[FALSE]);
	}

	/**
	 * Calculates probability of a node being true based on its markov blanket, i.e.
	 * MB(X) = alpha<P(X=true|PARENTS(X))*PRODUCT(CHILDREN(X)|PARENTS(CHILDREN(X))), 
	 *               P(X=false|PARENTS(X))*PRODUCT(CHILDREN(X)|PARENTS(CHILDREN(X)))>
	 * 
	 * @param nodeIndex The index of node in nodes[] to calculate probability for.
	 * @return The probability of this node being true (between 0 and 1).
	 */
	private double MBProb(int nodeIndex) {
		double p[] = {0, 0}, childProduct, childProb;
		boolean value;

		/* P(X=true|PARENTS(X)) */
		p[TRUE] = nodes[nodeIndex].conditionalProbability();
		/* P(X=false|PARENTS(X)) */
		p[FALSE] = 1 - p[TRUE];

		value = nodes[nodeIndex].value;
		
		/* PRODUCT(CHILDREN(X)|PARENTS(CHILDREN(X))) when X=true */
		nodes[nodeIndex].value = true;
		childProduct = 1;
		for (int i = 0; i < nodes[nodeIndex].children.length; i++) {
			childProb = nodes[nodeIndex].children[i].conditionalProbability();
			childProduct *= (nodes[nodeIndex].children[i].value == true) ? childProb : (1 - childProb);
		}
		p[TRUE] *= childProduct;
		
		/* PRODUCT(CHILDREN(X)|PARENTS(CHILDREN(X))) when X=false */
		nodes[nodeIndex].value = false;
		childProduct = 1;
		for (int i = 0; i < nodes[nodeIndex].children.length; i++) {
			childProb = nodes[nodeIndex].children[i].conditionalProbability();
			childProduct *= (nodes[nodeIndex].children[i].value == true) ? childProb : (1 - childProb);
		}
		p[FALSE] *= childProduct;

		nodes[nodeIndex].value = value;

		/* Return the normalised true value */
		return p[TRUE] / (p[TRUE] + p[FALSE]);
	}

	/**
	 * MCMC inference. Returns probability of query variable being true given the values of the evidence variables, 
	 * estimated based on the given total number of samples (see MCMC-ASK method from text book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes The indices of the evidence nodes.
	 * @param evidenceValues The values of the indexed evidence nodes.
	 * @param N The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the 
	 * 	evidence.
	 */
	public double MCMCask(int queryNode, int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {
		assert (queryNode < nodes.length);
		assert (indicesOfEvidenceNodes.length == evidenceValues.length);

		int count[] = {0, 0};
		
		/* Assign evidence nodes and initialise variable nodes with a random value */
		for (int i = 0; i < nodes.length; i++) {
			if (!isEvidence(i, indicesOfEvidenceNodes, evidenceValues)) nodes[i].value = rand.nextBoolean();
		}

		/* Take N samples, keeping a vector of sums of the query node being true or false */
		for (int i = 0; i < N; i++) {
			count[(nodes[queryNode].value == true) ? TRUE : FALSE]++;

			/* Sample every non-evidence node based on its markov blanket */
			for (int j = 0; j < nodes.length; j++) {
				if (!isEvidence(j, indicesOfEvidenceNodes, evidenceValues)) {
					nodes[j].value = (rand.nextDouble() < MBProb(j)) ? true : false;
				}
			}
		}

		/* Return the normalised true value */
		return count[TRUE] / (double)(count[TRUE] + count[FALSE]);
	}

	/** The main method, with some example method calls. */
	public static void main(String[] ops) {
		/*  Create network.*/
		BayesNet b = new BayesNet();

		/* Print out results of some non-inference calculations */
		double[] bluffProbabilities = b.calculateBBluffProbabilities(BBLUFF_EXAMPLES);
		System.out.println("When Bot B is cocky, it bluffs "
					+ String.format("%.2f", (bluffProbabilities[0] * 100)) + "% of the time.");
		System.out.println("When Bot B is not cocky, it bluffs "
					+ String.format("%.2f", (bluffProbabilities[1] * 100)) + "% of the time.");

		double bluffWinProb = b.calculateExactEventProbability(new boolean[] {
						true, true, true, false, false, true, false });
		System.out.println("The probability of Bot B winning on a cocky bluff "
						+ "(with bet) and both bots have bad hands (A dealt) is: "
						+ String.format("%.6f",bluffWinProb));

		/* Sample five states from joint distribution and print them */
		System.out.println();
		System.out.println("Prior Sampling:");
		String s;
		for (int j = 0; j < b.nodes.length; j++) {
			s = (b.nodes[j].name.length() < 8) ? b.nodes[j].name : b.nodes[j].name.substring(0, 7);
			System.out.print(s + "\t");
		}
		System.out.println();
		for (int i = 0; i < 5; i++) {
			b.priorSample();
			b.printState();
		}

		/* Print out results of some example queries based on rejection sampling. */
		System.out.println();
		System.out.println("Rejection Sampling:");

		// Probability of B.GoodHand given bet and A not win.
		s = String.format("%.4f", b.rejectionSampling(4, new int[] { 5, 6 }, 
					new boolean[] { true, false }, 10000));
		System.out.println("Probability of B.GoodHand given bet and A not win:\t" + s);

		// Probability of betting given a cocky
		s = String.format("%.4f", b.rejectionSampling(1, new int[] { 0 }, new boolean[] { true }, 10000));
		System.out.println("Probability of betting given a cocky:\t\t\t" + s);

		// Probability of B.Goodhand given B.Bluff and A.Deal
		s = String.format("%.4f", b.rejectionSampling(4, new int[] { 1, 2 }, 
					new boolean[] { true, true }, 10000));
		System.out.println("Probability of B.Goodhand given B.Bluff and A.Deal: \t" + s);

		
		/* Print out results of some example queries based on likelihood weighting. */
		System.out.println();
		System.out.println("Likelihood Weighting:");

		// Probability of B.GoodHand given bet and A not win.
		s = String.format("%.4f", b.likelihoodWeighting(4, new int[] { 5, 6 }, 
					new boolean[] { true, false }, 10000));
		System.out.println("Probability of B.GoodHand given bet and A not win:\t" + s);

		// Probability of betting given a cocky
		s = String.format("%.4f", b.likelihoodWeighting(1, new int[] { 0 }, new boolean[] { true }, 10000));
		System.out.println("Probability of betting given a cocky:\t\t\t" + s);

		// Probability of B.Goodhand given B.Bluff and A.Deal
		s = String.format("%.4f", b.likelihoodWeighting(4, new int[] { 1, 2 }, 
					new boolean[] { true, true }, 10000));
		System.out.println("Probability of B.Goodhand given B.Bluff and A.Deal: \t" + s);

		/* Print out results of some example queries based on markov chain monte carlo. */
		System.out.println();
		System.out.println("Markov Chain Monte Carlo:");

		// Probability of B.GoodHand given bet and A not win.
		s = String.format("%.4f", b.MCMCask(4, new int[] { 5, 6 }, 
					new boolean[] { true, false }, 100000));
		System.out.println("Probability of B.GoodHand given bet and A not win:\t" + s);

		// Probability of betting given a cocky
		s = String.format("%.4f", b.MCMCask(1, new int[] { 0 }, new boolean[] { true }, 10000));
		System.out.println("Probability of betting given a cocky:\t\t\t" + s);

		// Probability of B.Goodhand given B.Bluff and A.Deal
		s = String.format("%.4f", b.MCMCask(4, new int[] { 1, 2 }, 
					new boolean[] { true, true }, 10000));
		System.out.println("Probability of B.Goodhand given B.Bluff and A.Deal: \t" + s);
	}
}


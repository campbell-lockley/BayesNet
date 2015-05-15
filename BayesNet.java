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
		private double[] probs;		// The probabilities for the CPT
		public boolean value;		// The current value of the node

		/** Initializes the node. */
		private Node(String n, Node[] pa, double[] pr) {
			name = n;
			parents = pa;
			probs = pr;
		}

		/**
		 * Returns conditional probability of value "true" for the current node
		 * based on the values of the parent nodes.
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

	/* A collection of examples describing whether Bot B is { cocky, bluffing } */
	public static final boolean[][] BBLUFF_EXAMPLES = { { true,  true  },
			{ true,  true  }, { true,  true  }, { true,  false },
			{ true,  true  }, { false, false }, { false, false },
			{ false, true  }, { false, false }, { false, false },
			{ false, false }, { false, false }, { false, true  } };

	/** Constructor that sets up the Poker-game network. */
	public BayesNet() {
		nodes = new Node[7];
		nodes[0] = new Node("B.Cocky", new Node[] {}, new double[] { 0.05 });
		nodes[1] = new Node("B.Bluff", new Node[] { nodes[0] },
				calculateBBluffProbabilities(BBLUFF_EXAMPLES));
		nodes[2] = new Node("A.Deals", new Node[] {}, new double[] { 0.5 });
		nodes[3] = new Node("A.GoodHand", new Node[] { nodes[2] }, new double[] { 0.75, 0.5 });
		nodes[4] = new Node("B.GoodHand", new Node[] { nodes[2] }, new double[] { 0.4, 0.5 });
		nodes[5] = new Node("B.Bets", new Node[] { nodes[1], nodes[4] },
				new double[] { 0.95, 0.7, 0.9, 0.01 });
		nodes[6] = new Node("A.Wins", new Node[] { nodes[3], nodes[4] },
				new double[] { 0.45, 0.75, 0.25, 0.55 });
	}

	/** Prints the current state of the network to standard out. */
	public void printState() {
		for (int i = 0; i < nodes.length; i++) {
			if (i > 0) System.out.print(", ");
			System.out.print(nodes[i].name + " = " + nodes[i].value);
		}
		System.out.println();
		System.out.flush();
	}

	/**
	 * Calculates the probability that Bot B will bluff based on whether it is
	 * cocky or not.
	 * 
	 * @param bluffInstances A set of training examples in the form { cocky, bluff } from
	 * 		which to compute the probabilities.
	 * @return The probability that Bot B will bluff when it is { cocky, !cocky }.
	 */
	public double[] calculateBBluffProbabilities(boolean[][] bluffInstances) {
		double[] probabilities = new double[2];
		final int COCKY = 0, BLUFF = 1;
		int total = bluffInstances.length;
		int numCocky = 0;

		/* Calculate conditional probability table for B.Bluff using trainging data in BBLUFF_EXAMPLES */
		for (int i = 0; i < bluffInstances.length; i++) {
			if (bluffInstances[i][COCKY]) numCocky++;			// Find P(Cocky)

			if (bluffInstances[i][BLUFF]) {
				if (bluffInstances[i][COCKY]) probabilities[0]++;	// Find P(Bluff^Cocky)
				else probabilities[1]++;				// Find P(Bluff^~Cocky)
			}
		}
		probabilities[0] = probabilities[0] / numCocky;				// Calc. P(Bluff|Cocky)
		probabilities[1] = probabilities[1] / (total - numCocky);		// Calc. P(Bluff|~Cocky)

		return probabilities;
	}

	/**
	 * This method calculates the exact probability of a given event occurring,
	 * where all variables are assigned a given evidence value.
	 *
	 * @param evidenceValues The values of all nodes.
	 * @return -1 if the evidence does not cover every node in the network.
	 *         Otherwise a probability between 0 and 1.
	 */
	public double calculateExactEventProbability(boolean[] evidenceValues) {
		/* Only performs exact calculation for all evidence known. */
		if (evidenceValues.length != nodes.length) return -1;

		double prob = 1, nodeProb;

		/* Update bayesian network with current evidence */
		for (int i = 0; i < evidenceValues.length; i++) nodes[i].value = evidenceValues[i];

		/* P(x1, ..., xn) = P(x1|PARENTS(x1)) x ... x P(xn|PARENTS(xn)) */
		for (int i = 0; i < evidenceValues.length; i++) {
			nodeProb = nodes[i].conditionalProbability();
			prob *= (evidenceValues[i]) ? nodeProb : (1 - nodeProb);
		}

		return prob;
	}

	/**
	 * This method assigns new values to the nodes in the network by sampling
	 * from the joint distribution (based on PRIOR-SAMPLE method from text
	 * book/slides).
	 */
	public void priorSample() {

		// YOUR CODE HERE
	}

	/**
	 * Rejection sampling. Returns probability of query variable being true
	 * given the values of the evidence variables, estimated based on the given
	 * total number of samples (see REJECTION-SAMPLING method from text
	 * book/slides).
	 * 
	 * The nodes/variables are specified by their indices in the nodes array.
	 * The array evidenceValues has one value for each index in
	 * indicesOfEvidenceNodes. See also examples in main().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double rejectionSampling(int queryNode,
			int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/**
	 * This method assigns new values to the non-evidence nodes in the network
	 * and computes a weight based on the evidence nodes (based on
	 * WEIGHTED-SAMPLE method from text book/slides).
	 * 
	 * The evidence is specified as in the case of rejectionSampling().
	 * 
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @return The weight of the event occurring.
	 * 
	 */
	public double weightedSample(int[] indicesOfEvidenceNodes,
			boolean[] evidenceValues) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/**
	 * Likelihood weighting. Returns probability of query variable being true
	 * given the values of the evidence variables, estimated based on the given
	 * total number of samples (see LIKELIHOOD-WEIGHTING method from text
	 * book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double likelihoodWeighting(int queryNode,
			int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/**
	 * MCMC inference. Returns probability of query variable being true given
	 * the values of the evidence variables, estimated based on the given total
	 * number of samples (see MCMC-ASK method from text book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double MCMCask(int queryNode, int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/** The main method, with some example method calls. */
	public static void main(String[] ops) {
		/*  Create network.*/
		BayesNet b = new BayesNet();

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

		// Sample five states from joint distribution and print them
		for (int i = 0; i < 5; i++) {
			b.priorSample();
			b.printState();
		}

		// Print out results of some example queries based on rejection
		// sampling.
		// Same should be possible with likelihood weighting and MCMC inference.

		// Probability of B.GoodHand given bet and A not win.
		System.out.println(b.rejectionSampling(4, new int[] { 5, 6 }, new boolean[] { true, false }, 10000));

		// Probability of betting given a cocky
		System.out.println(b.rejectionSampling(1, new int[] { 0 }, new boolean[] { true }, 10000));

		// Probability of B.Goodhand given B.Bluff and A.Deal
		System.out.println(b.rejectionSampling(4, new int[] { 1, 2 }, new boolean[] { true, true }, 10000));
	}
}


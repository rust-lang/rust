use crate::fx::FxHashMap;
use std::cmp::max;
use std::slice;
use std::iter;

use super::*;

pub struct TestGraph {
    num_nodes: usize,
    start_node: usize,
    successors: FxHashMap<usize, Vec<usize>>,
    predecessors: FxHashMap<usize, Vec<usize>>,
}

impl TestGraph {
    pub fn new(start_node: usize, edges: &[(usize, usize)]) -> Self {
        let mut graph = TestGraph {
            num_nodes: start_node + 1,
            start_node,
            successors: FxHashMap::default(),
            predecessors: FxHashMap::default(),
        };
        for &(source, target) in edges {
            graph.num_nodes = max(graph.num_nodes, source + 1);
            graph.num_nodes = max(graph.num_nodes, target + 1);
            graph.successors.entry(source).or_default().push(target);
            graph.predecessors.entry(target).or_default().push(source);
        }
        for node in 0..graph.num_nodes {
            graph.successors.entry(node).or_default();
            graph.predecessors.entry(node).or_default();
        }
        graph
    }
}

impl DirectedGraph for TestGraph {
    type Node = usize;
}

impl WithStartNode for TestGraph {
    fn start_node(&self) -> usize {
        self.start_node
    }
}

impl WithNumNodes for TestGraph {
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl WithPredecessors for TestGraph {
    fn predecessors(&self,
                    node: usize)
                    -> <Self as GraphPredecessors<'_>>::Iter {
        self.predecessors[&node].iter().cloned()
    }
}

impl WithSuccessors for TestGraph {
    fn successors(&self, node: usize) -> <Self as GraphSuccessors<'_>>::Iter {
        self.successors[&node].iter().cloned()
    }
}

impl<'graph> GraphPredecessors<'graph> for TestGraph {
    type Item = usize;
    type Iter = iter::Cloned<slice::Iter<'graph, usize>>;
}

impl<'graph> GraphSuccessors<'graph> for TestGraph {
    type Item = usize;
    type Iter = iter::Cloned<slice::Iter<'graph, usize>>;
}

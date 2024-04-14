use crate::fx::FxHashMap;
use std::cmp::max;
use std::iter;
use std::slice;

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

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl StartNode for TestGraph {
    fn start_node(&self) -> usize {
        self.start_node
    }
}

impl Predecessors for TestGraph {
    type Predecessors<'g> = iter::Cloned<slice::Iter<'g, usize>>;

    fn predecessors(&self, node: usize) -> Self::Predecessors<'_> {
        self.predecessors[&node].iter().cloned()
    }
}

impl Successors for TestGraph {
    type Successors<'g> = iter::Cloned<slice::Iter<'g, usize>>;

    fn successors(&self, node: usize) -> Self::Successors<'_> {
        self.successors[&node].iter().cloned()
    }
}

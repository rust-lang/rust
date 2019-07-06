use super::*;

impl<'graph, G: DirectedGraph> DirectedGraph for &'graph G {
    type Node = G::Node;
}

impl<'graph, G: WithNumNodes> WithNumNodes for &'graph G {
    fn num_nodes(&self) -> usize {
        (**self).num_nodes()
    }
}

impl<'graph, G: WithStartNode> WithStartNode for &'graph G {
    fn start_node(&self) -> Self::Node {
        (**self).start_node()
    }
}

impl<'graph, G: WithSuccessors> WithSuccessors for &'graph G {
    fn successors(&self, node: Self::Node) -> <Self as GraphSuccessors<'_>>::Iter {
        (**self).successors(node)
    }
}

impl<'graph, G: WithPredecessors> WithPredecessors for &'graph G {
    fn predecessors(&self,
                    node: Self::Node)
                    -> <Self as GraphPredecessors<'_>>::Iter {
        (**self).predecessors(node)
    }
}

impl<'iter, 'graph, G: WithPredecessors> GraphPredecessors<'iter> for &'graph G {
    type Item = G::Node;
    type Iter = <G as GraphPredecessors<'iter>>::Iter;
}

impl<'iter, 'graph, G: WithSuccessors> GraphSuccessors<'iter> for &'graph G {
    type Item = G::Node;
    type Iter = <G as GraphSuccessors<'iter>>::Iter;
}

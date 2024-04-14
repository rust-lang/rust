use super::*;

impl<'graph, G: DirectedGraph> DirectedGraph for &'graph G {
    type Node = G::Node;

    fn num_nodes(&self) -> usize {
        (**self).num_nodes()
    }
}

impl<'graph, G: WithStartNode> WithStartNode for &'graph G {
    fn start_node(&self) -> Self::Node {
        (**self).start_node()
    }
}

impl<'graph, G: Successors> Successors for &'graph G {
    type Successors<'g> = G::Successors<'g> where 'graph: 'g;

    fn successors(&self, node: Self::Node) -> Self::Successors<'_> {
        (**self).successors(node)
    }
}

impl<'graph, G: Predecessors> Predecessors for &'graph G {
    type Predecessors<'g> = G::Predecessors<'g> where 'graph: 'g;

    fn predecessors(&self, node: Self::Node) -> Self::Predecessors<'_> {
        (**self).predecessors(node)
    }
}

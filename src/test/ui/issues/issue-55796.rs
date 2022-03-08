pub trait EdgeTrait<N> {
    fn target(&self) -> N;
}

pub trait Graph<'a> {
    type Node;
    type Edge: EdgeTrait<Self::Node>;
    type NodesIter: Iterator<Item = Self::Node> + 'a;
    type EdgesIter: Iterator<Item = Self::Edge> + 'a;

    fn nodes(&'a self) -> Self::NodesIter;
    fn out_edges(&'a self, u: &Self::Node) -> Self::EdgesIter;
    fn in_edges(&'a self, u: &Self::Node) -> Self::EdgesIter;

    fn out_neighbors(&'a self, u: &Self::Node) -> Box<dyn Iterator<Item = Self::Node>> {
        Box::new(self.out_edges(u).map(|e| e.target()))
        //~^ ERROR cannot infer
    }

    fn in_neighbors(&'a self, u: &Self::Node) -> Box<dyn Iterator<Item = Self::Node>> {
        Box::new(self.in_edges(u).map(|e| e.target()))
        //~^ ERROR cannot infer
    }
}

fn main() {}

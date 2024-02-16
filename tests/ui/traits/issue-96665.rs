//@ check-pass

pub trait Sequence<Item, Subsequence: Sequence<Item, Subsequence>> {}

pub trait NodeWalk<Graph: GraphBase, NodeSubwalk: NodeWalk<Graph, NodeSubwalk>>:
    Sequence<Graph::NodeIndex, NodeSubwalk>
{
}

pub trait GraphBase {
    type NodeIndex;
}

pub trait WalkableGraph: GraphBase {}

fn main() {}

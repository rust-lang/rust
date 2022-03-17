pub trait DirectedGraph {}
pub trait WithStartNode {}
pub trait WithPredecessors {}
pub trait WithSuccessors {}
pub trait WithNumNodes {}

pub trait ControlFlowGraph:
    DirectedGraph + WithStartNode + WithPredecessors + WithStartNode + WithSuccessors + WithNumNodes
                                                       //~ ERROR duplicate trait bound
{}

fn main() {}

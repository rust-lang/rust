use rustc_data_structures::graph::NodeIndex;
use rustc_data_structures::unify::UnifyKey;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DagId {
    index: u32,
}

impl DagId {
    pub fn from_input_index(n: NodeIndex) -> Self {
        DagId { index: n.0 as u32 }
    }

    pub fn as_input_index(&self) -> NodeIndex {
        NodeIndex(self.index as usize)
    }
}

impl UnifyKey for DagId {
    type Value = ();

    fn index(&self) -> u32 {
        self.index
    }

    fn from_index(u: u32) -> Self {
        DagId { index: u }
    }

    fn tag(_: Option<Self>) -> &'static str {
        "DagId"
    }
}

use super::ScalarInt;
use rustc_macros::HashStable;

#[derive(Clone, Debug, Hash, TyEncodable, TyDecodable, Eq, PartialEq, Ord, PartialOrd)]
#[derive(HashStable)]
pub enum ValTree {
    Leaf(ScalarInt),
    Branch(Vec<ValTree>),
}

impl ValTree {
    pub fn zst() -> Self {
        Self::Branch(Vec::new())
    }
}

use super::ScalarInt;
use rustc_macros::HashStable;

#[derive(Copy, Clone, Debug, Hash, TyEncodable, TyDecodable, Eq, PartialEq, Ord, PartialOrd)]
#[derive(HashStable)]
pub enum ValTree<'tcx> {
    Leaf(ScalarInt),
    Branch(&'tcx [ValTree<'tcx>]),
}

impl ValTree<'tcx> {
    pub fn zst() -> Self {
        Self::Branch(&[])
    }
}

use crate::ty::{self, VariantIdx};

/// The constituent parts of a type level constant of kind ADT or array.
#[derive(Copy, Clone, Debug, HashStable)]
pub struct DestructuredConst<'tcx> {
    pub variant: Option<VariantIdx>,
    pub fields: &'tcx [ty::Const<'tcx>],
}

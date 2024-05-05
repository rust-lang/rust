use rustc_hir::FnSig;

use crate::borrow_pats::SimpleTyKind;

use super::{AnalysisInfo, RETURN_LOCAL};

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, serde::Serialize)]
pub enum BodyPat {
    Borrow,
    /// This function doesn't take any arguments
    NoArguments,
    /// Indicates that a body contained an anonymous loan. These are usually
    /// only used for temp borrows.
    OwnedArgBorrow,
    OwnedArgBorrowMut,
    OwnedClosureBorrow,
    OwnedClosureBorrowMut,
    /// Indicates that a body contained a named loan. So cases like
    /// `_2 = &_1`, where `_2` is a named variable.
    OwnedNamedBorrow,
    OwnedNamedBorrowMut,
    /// This function uses a two phased borrow. This is different from rustc's
    /// value tracked in `BorrowKind` as this this pattern is only added if a two
    /// phase borrow actually happened (i.e. the code would be rejected without)
    OwnedTwoPhaseBorrow,
    ReturnedStaticLoanForNonStatic,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, serde::Serialize)]
pub struct BodyInfo {
    pub(super) return_ty: SimpleTyKind,
    pub(super) is_const: bool,
    pub(super) is_safe: bool,
    pub(super) is_sync: bool,
    pub(super) context: BodyContext,
}

impl BodyInfo {
    pub fn from_sig(hir_sig: &FnSig<'_>, info: &AnalysisInfo<'_>, context: BodyContext) -> Self {
        Self {
            return_ty: SimpleTyKind::from_ty(info.body.local_decls[RETURN_LOCAL].ty),
            is_const: hir_sig.header.is_const(),
            is_safe: !hir_sig.header.is_unsafe(),
            is_sync: !hir_sig.header.is_async(),
            context,
        }
    }
}

impl std::fmt::Debug for BodyInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BodyInfo({self})")
    }
}

impl std::fmt::Display for BodyInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let return_ty = format!("Output({:?})", self.return_ty);
        let constness = if self.is_const { "Const" } else { "NotConst" };
        let safety = if self.is_safe { "Safe" } else { "Unsafe" };
        let sync = if self.is_sync { "Sync" } else { "Async" };
        let context = format!("{:?}", self.context);
        write!(
            f,
            "{return_ty:<17}, {constness:<9}, {safety:<6}, {sync:<5}, {context:<10}"
        )
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, serde::Serialize)]
pub enum BodyContext {
    /// The function is simple and free.
    Free,
    /// The function is inside an `impl` block.
    Impl,
    /// The function is inside an `impl Trait` block.
    TraitImpl,
    /// The function is inside a trait definition.
    TraitDef,
}

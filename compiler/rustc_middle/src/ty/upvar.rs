use crate::ty;

use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_span::Span;

use super::BorrowKind;

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    TyEncodable,
    TyDecodable,
    TypeFoldable,
    HashStable
)]
pub struct UpvarPath {
    pub hir_id: hir::HirId,
}

/// Upvars do not get their own `NodeId`. Instead, we use the pair of
/// the original var ID (that is, the root variable that is referenced
/// by the upvar) and the ID of the closure expression.
#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable, TypeFoldable, HashStable)]
pub struct UpvarId {
    pub var_path: UpvarPath,
    pub closure_expr_id: LocalDefId,
}

impl UpvarId {
    pub fn new(var_hir_id: hir::HirId, closure_def_id: LocalDefId) -> UpvarId {
        UpvarId { var_path: UpvarPath { hir_id: var_hir_id }, closure_expr_id: closure_def_id }
    }
}

/// Information describing the capture of an upvar. This is computed
/// during `typeck`, specifically by `regionck`.
#[derive(PartialEq, Clone, Debug, Copy, TyEncodable, TyDecodable, TypeFoldable, HashStable)]
pub enum UpvarCapture<'tcx> {
    /// Upvar is captured by value. This is always true when the
    /// closure is labeled `move`, but can also be true in other cases
    /// depending on inference.
    ///
    /// If the upvar was inferred to be captured by value (e.g. `move`
    /// was not used), then the `Span` points to a usage that
    /// required it. There may be more than one such usage
    /// (e.g. `|| { a; a; }`), in which case we pick an
    /// arbitrary one.
    ByValue(Option<Span>),

    /// Upvar is captured by reference.
    ByRef(UpvarBorrow<'tcx>),
}

#[derive(PartialEq, Clone, Copy, TyEncodable, TyDecodable, TypeFoldable, HashStable)]
pub struct UpvarBorrow<'tcx> {
    /// The kind of borrow: by-ref upvars have access to shared
    /// immutable borrows, which are not part of the normal language
    /// syntax.
    pub kind: BorrowKind,

    /// Region of the resulting reference.
    pub region: ty::Region<'tcx>,
}

pub type UpvarListMap = FxHashMap<DefId, FxIndexMap<hir::HirId, UpvarId>>;
pub type UpvarCaptureMap<'tcx> = FxHashMap<UpvarId, UpvarCapture<'tcx>>;

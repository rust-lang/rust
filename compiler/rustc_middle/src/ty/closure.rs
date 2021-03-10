use crate::hir::place::{Place as HirPlace, PlaceBase as HirPlaceBase};
use crate::ty;

use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_span::Span;

use super::{BorrowKind, CaptureInfo, Ty, TyCtxt};

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

/// Given the closure DefId this map provides a map of root variables to minimum
/// set of `CapturedPlace`s that need to be tracked to support all captures of that closure.
pub type MinCaptureInformationMap<'tcx> = FxHashMap<DefId, RootVariableMinCaptureList<'tcx>>;

/// Part of `MinCaptureInformationMap`; Maps a root variable to the list of `CapturedPlace`.
/// Used to track the minimum set of `Place`s that need to be captured to support all
/// Places captured by the closure starting at a given root variable.
///
/// This provides a convenient and quick way of checking if a variable being used within
/// a closure is a capture of a local variable.
pub type RootVariableMinCaptureList<'tcx> = FxIndexMap<hir::HirId, MinCaptureList<'tcx>>;

/// Part of `MinCaptureInformationMap`; List of `CapturePlace`s.
pub type MinCaptureList<'tcx> = Vec<CapturedPlace<'tcx>>;

/// Represents the various closure traits in the language. This
/// will determine the type of the environment (`self`, in the
/// desugaring) argument that the closure expects.
///
/// You can get the environment type of a closure using
/// `tcx.closure_env_ty()`.
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub enum ClosureKind {
    // Warning: Ordering is significant here! The ordering is chosen
    // because the trait Fn is a subtrait of FnMut and so in turn, and
    // hence we order it so that Fn < FnMut < FnOnce.
    Fn,
    FnMut,
    FnOnce,
}

impl<'tcx> ClosureKind {
    // This is the initial value used when doing upvar inference.
    pub const LATTICE_BOTTOM: ClosureKind = ClosureKind::Fn;

    pub fn trait_did(&self, tcx: TyCtxt<'tcx>) -> DefId {
        match *self {
            ClosureKind::Fn => tcx.require_lang_item(LangItem::Fn, None),
            ClosureKind::FnMut => tcx.require_lang_item(LangItem::FnMut, None),
            ClosureKind::FnOnce => tcx.require_lang_item(LangItem::FnOnce, None),
        }
    }

    /// Returns `true` if a type that impls this closure kind
    /// must also implement `other`.
    pub fn extends(self, other: ty::ClosureKind) -> bool {
        matches!(
            (self, other),
            (ClosureKind::Fn, ClosureKind::Fn)
                | (ClosureKind::Fn, ClosureKind::FnMut)
                | (ClosureKind::Fn, ClosureKind::FnOnce)
                | (ClosureKind::FnMut, ClosureKind::FnMut)
                | (ClosureKind::FnMut, ClosureKind::FnOnce)
                | (ClosureKind::FnOnce, ClosureKind::FnOnce)
        )
    }

    /// Returns the representative scalar type for this closure kind.
    /// See `TyS::to_opt_closure_kind` for more details.
    pub fn to_ty(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self {
            ty::ClosureKind::Fn => tcx.types.i8,
            ty::ClosureKind::FnMut => tcx.types.i16,
            ty::ClosureKind::FnOnce => tcx.types.i32,
        }
    }
}

/// A composite describing a `Place` that is captured by a closure.
#[derive(PartialEq, Clone, Debug, TyEncodable, TyDecodable, TypeFoldable, HashStable)]
pub struct CapturedPlace<'tcx> {
    /// The `Place` that is captured.
    pub place: HirPlace<'tcx>,

    /// `CaptureKind` and expression(s) that resulted in such capture of `place`.
    pub info: CaptureInfo<'tcx>,

    /// Represents if `place` can be mutated or not.
    pub mutability: hir::Mutability,
}

impl CapturedPlace<'tcx> {
    /// Returns the hir-id of the root variable for the captured place.
    /// e.g., if `a.b.c` was captured, would return the hir-id for `a`.
    pub fn get_root_variable(&self) -> hir::HirId {
        match self.place.base {
            HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
            base => bug!("Expected upvar, found={:?}", base),
        }
    }
}

use crate::hir::place::{
    Place as HirPlace, PlaceBase as HirPlaceBase, ProjectionKind as HirProjectionKind,
};
use crate::{mir, ty};

use std::fmt::Write;

use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_span::{Span, Symbol};

use super::{Ty, TyCtxt};

use self::BorrowKind::*;

// Captures are represented using fields inside a structure.
// This represents accessing self in the closure structure
pub const CAPTURE_STRUCT_LOCAL: mir::Local = mir::Local::from_u32(1);

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
pub enum UpvarCapture {
    /// Upvar is captured by value. This is always true when the
    /// closure is labeled `move`, but can also be true in other cases
    /// depending on inference.
    ByValue,

    /// Upvar is captured by reference.
    ByRef(BorrowKind),
}

pub type UpvarListMap = FxHashMap<DefId, FxIndexMap<hir::HirId, UpvarId>>;
pub type UpvarCaptureMap = FxHashMap<UpvarId, UpvarCapture>;

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
    /// See `Ty::to_opt_closure_kind` for more details.
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
    pub info: CaptureInfo,

    /// Represents if `place` can be mutated or not.
    pub mutability: hir::Mutability,

    /// Region of the resulting reference if the upvar is captured by ref.
    pub region: Option<ty::Region<'tcx>>,
}

impl<'tcx> CapturedPlace<'tcx> {
    pub fn to_string(&self, tcx: TyCtxt<'tcx>) -> String {
        place_to_string_for_capture(tcx, &self.place)
    }

    /// Returns a symbol of the captured upvar, which looks like `name__field1__field2`.
    fn to_symbol(&self, tcx: TyCtxt<'tcx>) -> Symbol {
        let hir_id = match self.place.base {
            HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
            base => bug!("Expected an upvar, found {:?}", base),
        };
        let mut symbol = tcx.hir().name(hir_id).as_str().to_string();

        let mut ty = self.place.base_ty;
        for proj in self.place.projections.iter() {
            match proj.kind {
                HirProjectionKind::Field(idx, variant) => match ty.kind() {
                    ty::Tuple(_) => write!(&mut symbol, "__{}", idx).unwrap(),
                    ty::Adt(def, ..) => {
                        write!(
                            &mut symbol,
                            "__{}",
                            def.variants[variant].fields[idx as usize].name.as_str(),
                        )
                        .unwrap();
                    }
                    ty => {
                        bug!("Unexpected type {:?} for `Field` projection", ty)
                    }
                },

                // Ignore derefs for now, as they are likely caused by
                // autoderefs that don't appear in the original code.
                HirProjectionKind::Deref => {}
                proj => bug!("Unexpected projection {:?} in captured place", proj),
            }
            ty = proj.ty;
        }

        Symbol::intern(&symbol)
    }

    /// Returns the hir-id of the root variable for the captured place.
    /// e.g., if `a.b.c` was captured, would return the hir-id for `a`.
    pub fn get_root_variable(&self) -> hir::HirId {
        match self.place.base {
            HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
            base => bug!("Expected upvar, found={:?}", base),
        }
    }

    /// Returns the `LocalDefId` of the closure that captured this Place
    pub fn get_closure_local_def_id(&self) -> LocalDefId {
        match self.place.base {
            HirPlaceBase::Upvar(upvar_id) => upvar_id.closure_expr_id,
            base => bug!("expected upvar, found={:?}", base),
        }
    }

    /// Return span pointing to use that resulted in selecting the captured path
    pub fn get_path_span(&self, tcx: TyCtxt<'tcx>) -> Span {
        if let Some(path_expr_id) = self.info.path_expr_id {
            tcx.hir().span(path_expr_id)
        } else if let Some(capture_kind_expr_id) = self.info.capture_kind_expr_id {
            tcx.hir().span(capture_kind_expr_id)
        } else {
            // Fallback on upvars mentioned if neither path or capture expr id is captured

            // Safe to unwrap since we know this place is captured by the closure, therefore the closure must have upvars.
            tcx.upvars_mentioned(self.get_closure_local_def_id()).unwrap()
                [&self.get_root_variable()]
                .span
        }
    }

    /// Return span pointing to use that resulted in selecting the current capture kind
    pub fn get_capture_kind_span(&self, tcx: TyCtxt<'tcx>) -> Span {
        if let Some(capture_kind_expr_id) = self.info.capture_kind_expr_id {
            tcx.hir().span(capture_kind_expr_id)
        } else if let Some(path_expr_id) = self.info.path_expr_id {
            tcx.hir().span(path_expr_id)
        } else {
            // Fallback on upvars mentioned if neither path or capture expr id is captured

            // Safe to unwrap since we know this place is captured by the closure, therefore the closure must have upvars.
            tcx.upvars_mentioned(self.get_closure_local_def_id()).unwrap()
                [&self.get_root_variable()]
                .span
        }
    }
}

fn symbols_for_closure_captures<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: (LocalDefId, DefId),
) -> Vec<Symbol> {
    let typeck_results = tcx.typeck(def_id.0);
    let captures = typeck_results.closure_min_captures_flattened(def_id.1);
    captures.into_iter().map(|captured_place| captured_place.to_symbol(tcx)).collect()
}

/// Return true if the `proj_possible_ancestor` represents an ancestor path
/// to `proj_capture` or `proj_possible_ancestor` is same as `proj_capture`,
/// assuming they both start off of the same root variable.
///
/// **Note:** It's the caller's responsibility to ensure that both lists of projections
///           start off of the same root variable.
///
/// Eg: 1. `foo.x` which is represented using `projections=[Field(x)]` is an ancestor of
///        `foo.x.y` which is represented using `projections=[Field(x), Field(y)]`.
///        Note both `foo.x` and `foo.x.y` start off of the same root variable `foo`.
///     2. Since we only look at the projections here function will return `bar.x` as an a valid
///        ancestor of `foo.x.y`. It's the caller's responsibility to ensure that both projections
///        list are being applied to the same root variable.
pub fn is_ancestor_or_same_capture(
    proj_possible_ancestor: &[HirProjectionKind],
    proj_capture: &[HirProjectionKind],
) -> bool {
    // We want to make sure `is_ancestor_or_same_capture("x.0.0", "x.0")` to return false.
    // Therefore we can't just check if all projections are same in the zipped iterator below.
    if proj_possible_ancestor.len() > proj_capture.len() {
        return false;
    }

    proj_possible_ancestor.iter().zip(proj_capture).all(|(a, b)| a == b)
}

/// Part of `MinCaptureInformationMap`; describes the capture kind (&, &mut, move)
/// for a particular capture as well as identifying the part of the source code
/// that triggered this capture to occur.
#[derive(PartialEq, Clone, Debug, Copy, TyEncodable, TyDecodable, TypeFoldable, HashStable)]
pub struct CaptureInfo {
    /// Expr Id pointing to use that resulted in selecting the current capture kind
    ///
    /// Eg:
    /// ```rust,no_run
    /// let mut t = (0,1);
    ///
    /// let c = || {
    ///     println!("{}",t); // L1
    ///     t.1 = 4; // L2
    /// };
    /// ```
    /// `capture_kind_expr_id` will point to the use on L2 and `path_expr_id` will point to the
    /// use on L1.
    ///
    /// If the user doesn't enable feature `capture_disjoint_fields` (RFC 2229) then, it is
    /// possible that we don't see the use of a particular place resulting in capture_kind_expr_id being
    /// None. In such case we fallback on uvpars_mentioned for span.
    ///
    /// Eg:
    /// ```rust,no_run
    /// let x = 5;
    ///
    /// let c = || {
    ///     let _ = x
    /// };
    /// ```
    ///
    /// In this example, if `capture_disjoint_fields` is **not** set, then x will be captured,
    /// but we won't see it being used during capture analysis, since it's essentially a discard.
    pub capture_kind_expr_id: Option<hir::HirId>,
    /// Expr Id pointing to use that resulted the corresponding place being captured
    ///
    /// See `capture_kind_expr_id` for example.
    ///
    pub path_expr_id: Option<hir::HirId>,

    /// Capture mode that was selected
    pub capture_kind: UpvarCapture,
}

pub fn place_to_string_for_capture<'tcx>(tcx: TyCtxt<'tcx>, place: &HirPlace<'tcx>) -> String {
    let mut curr_string: String = match place.base {
        HirPlaceBase::Upvar(upvar_id) => tcx.hir().name(upvar_id.var_path.hir_id).to_string(),
        _ => bug!("Capture_information should only contain upvars"),
    };

    for (i, proj) in place.projections.iter().enumerate() {
        match proj.kind {
            HirProjectionKind::Deref => {
                curr_string = format!("*{}", curr_string);
            }
            HirProjectionKind::Field(idx, variant) => match place.ty_before_projection(i).kind() {
                ty::Adt(def, ..) => {
                    curr_string = format!(
                        "{}.{}",
                        curr_string,
                        def.variants[variant].fields[idx as usize].name.as_str()
                    );
                }
                ty::Tuple(_) => {
                    curr_string = format!("{}.{}", curr_string, idx);
                }
                _ => {
                    bug!(
                        "Field projection applied to a type other than Adt or Tuple: {:?}.",
                        place.ty_before_projection(i).kind()
                    )
                }
            },
            proj => bug!("{:?} unexpected because it isn't captured", proj),
        }
    }

    curr_string
}

#[derive(Clone, PartialEq, Debug, TyEncodable, TyDecodable, TypeFoldable, Copy, HashStable)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    ImmBorrow,

    /// Data must be immutable but not aliasable. This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when the closure
    /// is borrowing or mutating a mutable referent, e.g.:
    ///
    /// ```
    /// let x: &mut isize = ...;
    /// let y = || *x += 5;
    /// ```
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    /// ```
    /// struct Env { x: & &mut isize }
    /// let x: &mut isize = ...;
    /// let y = (&mut Env { &x }, fn_ptr);  // Closure is pair of env and fn
    /// fn fn_ptr(env: &mut Env) { **env.x += 5; }
    /// ```
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    /// ```
    /// struct Env { x: &mut &mut isize }
    /// let x: &mut isize = ...;
    /// let y = (&mut Env { &mut x }, fn_ptr); // changed from &x to &mut x
    /// fn fn_ptr(env: &mut Env) { **env.x += 5; }
    /// ```
    ///
    /// Now the assignment to `**env.x` is legal, but creating a
    /// mutable pointer to `x` is not because `x` is not mutable. We
    /// could fix this by declaring `x` as `let mut x`. This is ok in
    /// user code, if awkward, but extra weird for closures, since the
    /// borrow is hidden.
    ///
    /// So we introduce a "unique imm" borrow -- the referent is
    /// immutable, but not aliasable. This solves the problem. For
    /// simplicity, we don't give users the way to express this
    /// borrow, it's just used when translating closures.
    UniqueImmBorrow,

    /// Data is mutable and not aliasable.
    MutBorrow,
}

impl BorrowKind {
    pub fn from_mutbl(m: hir::Mutability) -> BorrowKind {
        match m {
            hir::Mutability::Mut => MutBorrow,
            hir::Mutability::Not => ImmBorrow,
        }
    }

    /// Returns a mutability `m` such that an `&m T` pointer could be used to obtain this borrow
    /// kind. Because borrow kinds are richer than mutabilities, we sometimes have to pick a
    /// mutability that is stronger than necessary so that it at least *would permit* the borrow in
    /// question.
    pub fn to_mutbl_lossy(self) -> hir::Mutability {
        match self {
            MutBorrow => hir::Mutability::Mut,
            ImmBorrow => hir::Mutability::Not,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of a `&uniq`
            // and hence is a safe "over approximation".
            UniqueImmBorrow => hir::Mutability::Mut,
        }
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { symbols_for_closure_captures, ..*providers }
}

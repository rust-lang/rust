use std::fmt::Write;

use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_hir::HirId;
use rustc_hir::def_id::LocalDefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::def_id::LocalDefIdMap;
use rustc_span::{Ident, Span, Symbol};

use super::TyCtxt;
use crate::hir::place::{
    Place as HirPlace, PlaceBase as HirPlaceBase, ProjectionKind as HirProjectionKind,
};
use crate::query::Providers;
use crate::{mir, ty};

/// Captures are represented using fields inside a structure.
/// This represents accessing self in the closure structure
pub const CAPTURE_STRUCT_LOCAL: mir::Local = mir::Local::from_u32(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct UpvarPath {
    pub hir_id: HirId,
}

/// Upvars do not get their own `NodeId`. Instead, we use the pair of
/// the original var ID (that is, the root variable that is referenced
/// by the upvar) and the ID of the closure expression.
#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct UpvarId {
    pub var_path: UpvarPath,
    pub closure_expr_id: LocalDefId,
}

impl UpvarId {
    pub fn new(var_hir_id: HirId, closure_def_id: LocalDefId) -> UpvarId {
        UpvarId { var_path: UpvarPath { hir_id: var_hir_id }, closure_expr_id: closure_def_id }
    }
}

/// Information describing the capture of an upvar. This is computed
/// during `typeck`, specifically by `regionck`.
#[derive(Eq, PartialEq, Clone, Debug, Copy, TyEncodable, TyDecodable, HashStable, Hash)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum UpvarCapture {
    /// Upvar is captured by value. This is always true when the
    /// closure is labeled `move`, but can also be true in other cases
    /// depending on inference.
    ByValue,

    /// Upvar is captured by use. This is true when the closure is labeled `use`.
    ByUse,

    /// Upvar is captured by reference.
    ByRef(BorrowKind),
}

/// Given the closure DefId this map provides a map of root variables to minimum
/// set of `CapturedPlace`s that need to be tracked to support all captures of that closure.
pub type MinCaptureInformationMap<'tcx> = LocalDefIdMap<RootVariableMinCaptureList<'tcx>>;

/// Part of `MinCaptureInformationMap`; Maps a root variable to the list of `CapturedPlace`.
/// Used to track the minimum set of `Place`s that need to be captured to support all
/// Places captured by the closure starting at a given root variable.
///
/// This provides a convenient and quick way of checking if a variable being used within
/// a closure is a capture of a local variable.
pub type RootVariableMinCaptureList<'tcx> = FxIndexMap<HirId, MinCaptureList<'tcx>>;

/// Part of `MinCaptureInformationMap`; List of `CapturePlace`s.
pub type MinCaptureList<'tcx> = Vec<CapturedPlace<'tcx>>;

/// A composite describing a `Place` that is captured by a closure.
#[derive(Eq, PartialEq, Clone, Debug, TyEncodable, TyDecodable, HashStable, Hash)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct CapturedPlace<'tcx> {
    /// Name and span where the binding happens.
    pub var_ident: Ident,

    /// The `Place` that is captured.
    pub place: HirPlace<'tcx>,

    /// `CaptureKind` and expression(s) that resulted in such capture of `place`.
    pub info: CaptureInfo,

    /// Represents if `place` can be mutated or not.
    pub mutability: hir::Mutability,
}

impl<'tcx> CapturedPlace<'tcx> {
    pub fn to_string(&self, tcx: TyCtxt<'tcx>) -> String {
        place_to_string_for_capture(tcx, &self.place)
    }

    /// Returns a symbol of the captured upvar, which looks like `name__field1__field2`.
    pub fn to_symbol(&self) -> Symbol {
        let mut symbol = self.var_ident.to_string();

        let mut ty = self.place.base_ty;
        for proj in self.place.projections.iter() {
            match proj.kind {
                HirProjectionKind::Field(idx, variant) => match ty.kind() {
                    ty::Tuple(_) => write!(&mut symbol, "__{}", idx.index()).unwrap(),
                    ty::Adt(def, ..) => {
                        write!(
                            &mut symbol,
                            "__{}",
                            def.variant(variant).fields[idx].name.as_str(),
                        )
                        .unwrap();
                    }
                    ty => {
                        bug!("Unexpected type {:?} for `Field` projection", ty)
                    }
                },

                HirProjectionKind::UnwrapUnsafeBinder => {
                    write!(&mut symbol, "__unwrap").unwrap();
                }

                // Ignore derefs for now, as they are likely caused by
                // autoderefs that don't appear in the original code.
                HirProjectionKind::Deref => {}
                // Just change the type to the hidden type, so we can actually project.
                HirProjectionKind::OpaqueCast => {}
                proj => bug!("Unexpected projection {:?} in captured place", proj),
            }
            ty = proj.ty;
        }

        Symbol::intern(&symbol)
    }

    /// Returns the hir-id of the root variable for the captured place.
    /// e.g., if `a.b.c` was captured, would return the hir-id for `a`.
    pub fn get_root_variable(&self) -> HirId {
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
            tcx.hir_span(path_expr_id)
        } else if let Some(capture_kind_expr_id) = self.info.capture_kind_expr_id {
            tcx.hir_span(capture_kind_expr_id)
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
            tcx.hir_span(capture_kind_expr_id)
        } else if let Some(path_expr_id) = self.info.path_expr_id {
            tcx.hir_span(path_expr_id)
        } else {
            // Fallback on upvars mentioned if neither path or capture expr id is captured

            // Safe to unwrap since we know this place is captured by the closure, therefore the closure must have upvars.
            tcx.upvars_mentioned(self.get_closure_local_def_id()).unwrap()
                [&self.get_root_variable()]
                .span
        }
    }

    pub fn is_by_ref(&self) -> bool {
        match self.info.capture_kind {
            ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => false,
            ty::UpvarCapture::ByRef(..) => true,
        }
    }
}

#[derive(Copy, Clone, Debug, HashStable)]
pub struct ClosureTypeInfo<'tcx> {
    user_provided_sig: ty::CanonicalPolyFnSig<'tcx>,
    captures: &'tcx ty::List<&'tcx ty::CapturedPlace<'tcx>>,
    kind_origin: Option<&'tcx (Span, HirPlace<'tcx>)>,
}

fn closure_typeinfo<'tcx>(tcx: TyCtxt<'tcx>, def: LocalDefId) -> ClosureTypeInfo<'tcx> {
    debug_assert!(tcx.is_closure_like(def.to_def_id()));
    let typeck_results = tcx.typeck(def);
    let user_provided_sig = typeck_results.user_provided_sigs[&def];
    let captures = typeck_results.closure_min_captures_flattened(def);
    let captures = tcx.mk_captures_from_iter(captures);
    let hir_id = tcx.local_def_id_to_hir_id(def);
    let kind_origin = typeck_results.closure_kind_origins().get(hir_id);
    ClosureTypeInfo { user_provided_sig, captures, kind_origin }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn closure_kind_origin(self, def_id: LocalDefId) -> Option<&'tcx (Span, HirPlace<'tcx>)> {
        self.closure_typeinfo(def_id).kind_origin
    }

    pub fn closure_user_provided_sig(self, def_id: LocalDefId) -> ty::CanonicalPolyFnSig<'tcx> {
        self.closure_typeinfo(def_id).user_provided_sig
    }

    pub fn closure_captures(self, def_id: LocalDefId) -> &'tcx [&'tcx ty::CapturedPlace<'tcx>] {
        if !self.is_closure_like(def_id.to_def_id()) {
            return &[];
        }
        self.closure_typeinfo(def_id).captures
    }
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
///     2. Since we only look at the projections here function will return `bar.x` as a valid
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
#[derive(Eq, PartialEq, Clone, Debug, Copy, TyEncodable, TyDecodable, HashStable, Hash)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct CaptureInfo {
    /// Expr Id pointing to use that resulted in selecting the current capture kind
    ///
    /// Eg:
    /// ```rust,no_run
    /// let mut t = (0,1);
    ///
    /// let c = || {
    ///     println!("{t:?}"); // L1
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
    ///     let _ = x;
    /// };
    /// ```
    ///
    /// In this example, if `capture_disjoint_fields` is **not** set, then x will be captured,
    /// but we won't see it being used during capture analysis, since it's essentially a discard.
    pub capture_kind_expr_id: Option<HirId>,
    /// Expr Id pointing to use that resulted the corresponding place being captured
    ///
    /// See `capture_kind_expr_id` for example.
    ///
    pub path_expr_id: Option<HirId>,

    /// Capture mode that was selected
    pub capture_kind: UpvarCapture,
}

pub fn place_to_string_for_capture<'tcx>(tcx: TyCtxt<'tcx>, place: &HirPlace<'tcx>) -> String {
    let mut curr_string: String = match place.base {
        HirPlaceBase::Upvar(upvar_id) => tcx.hir_name(upvar_id.var_path.hir_id).to_string(),
        _ => bug!("Capture_information should only contain upvars"),
    };

    for (i, proj) in place.projections.iter().enumerate() {
        match proj.kind {
            HirProjectionKind::Deref => {
                curr_string = format!("*{curr_string}");
            }
            HirProjectionKind::Field(idx, variant) => match place.ty_before_projection(i).kind() {
                ty::Adt(def, ..) => {
                    curr_string = format!(
                        "{}.{}",
                        curr_string,
                        def.variant(variant).fields[idx].name.as_str()
                    );
                }
                ty::Tuple(_) => {
                    curr_string = format!("{}.{}", curr_string, idx.index());
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

#[derive(Eq, Clone, PartialEq, Debug, TyEncodable, TyDecodable, Copy, HashStable, Hash)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    Immutable,

    /// Data must be immutable but not aliasable. This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when the closure
    /// is borrowing or mutating a mutable referent, e.g.:
    ///
    /// ```
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = || *x += 5;
    /// ```
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    /// ```compile_fail,E0594
    /// struct Env<'a> { x: &'a &'a mut isize }
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = (&mut Env { x: &x }, fn_ptr);  // Closure is pair of env and fn
    /// fn fn_ptr(env: &mut Env) { **env.x += 5; }
    /// ```
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    /// ```compile_fail,E0596
    /// struct Env<'a> { x: &'a mut &'a mut isize }
    /// let mut z = 3;
    /// let x: &mut isize = &mut z;
    /// let y = (&mut Env { x: &mut x }, fn_ptr); // changed from &x to &mut x
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
    ///
    /// FIXME: Rename this to indicate the borrow is actually not immutable.
    UniqueImmutable,

    /// Data is mutable and not aliasable.
    Mutable,
}

impl BorrowKind {
    pub fn from_mutbl(m: hir::Mutability) -> BorrowKind {
        match m {
            hir::Mutability::Mut => BorrowKind::Mutable,
            hir::Mutability::Not => BorrowKind::Immutable,
        }
    }

    /// Returns a mutability `m` such that an `&m T` pointer could be used to obtain this borrow
    /// kind. Because borrow kinds are richer than mutabilities, we sometimes have to pick a
    /// mutability that is stronger than necessary so that it at least *would permit* the borrow in
    /// question.
    pub fn to_mutbl_lossy(self) -> hir::Mutability {
        match self {
            BorrowKind::Mutable => hir::Mutability::Mut,
            BorrowKind::Immutable => hir::Mutability::Not,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of a `&uniq`
            // and hence is a safe "over approximation".
            BorrowKind::UniqueImmutable => hir::Mutability::Mut,
        }
    }
}

pub fn analyze_coroutine_closure_captures<'a, 'tcx: 'a, T>(
    parent_captures: impl IntoIterator<Item = &'a CapturedPlace<'tcx>>,
    child_captures: impl IntoIterator<Item = &'a CapturedPlace<'tcx>>,
    mut for_each: impl FnMut((usize, &'a CapturedPlace<'tcx>), (usize, &'a CapturedPlace<'tcx>)) -> T,
) -> impl Iterator<Item = T> {
    std::iter::from_coroutine(
        #[coroutine]
        move || {
            let mut child_captures = child_captures.into_iter().enumerate().peekable();

            // One parent capture may correspond to several child captures if we end up
            // refining the set of captures via edition-2021 precise captures. We want to
            // match up any number of child captures with one parent capture, so we keep
            // peeking off this `Peekable` until the child doesn't match anymore.
            for (parent_field_idx, parent_capture) in parent_captures.into_iter().enumerate() {
                // Make sure we use every field at least once, b/c why are we capturing something
                // if it's not used in the inner coroutine.
                let mut field_used_at_least_once = false;

                // A parent matches a child if they share the same prefix of projections.
                // The child may have more, if it is capturing sub-fields out of
                // something that is captured by-move in the parent closure.
                while child_captures.peek().is_some_and(|(_, child_capture)| {
                    child_prefix_matches_parent_projections(parent_capture, child_capture)
                }) {
                    let (child_field_idx, child_capture) = child_captures.next().unwrap();
                    // This analysis only makes sense if the parent capture is a
                    // prefix of the child capture.
                    assert!(
                        child_capture.place.projections.len()
                            >= parent_capture.place.projections.len(),
                        "parent capture ({parent_capture:#?}) expected to be prefix of \
                    child capture ({child_capture:#?})"
                    );

                    yield for_each(
                        (parent_field_idx, parent_capture),
                        (child_field_idx, child_capture),
                    );

                    field_used_at_least_once = true;
                }

                // Make sure the field was used at least once.
                assert!(
                    field_used_at_least_once,
                    "we captured {parent_capture:#?} but it was not used in the child coroutine?"
                );
            }
            assert_eq!(child_captures.next(), None, "leftover child captures?");
        },
    )
}

fn child_prefix_matches_parent_projections(
    parent_capture: &ty::CapturedPlace<'_>,
    child_capture: &ty::CapturedPlace<'_>,
) -> bool {
    let HirPlaceBase::Upvar(parent_base) = parent_capture.place.base else {
        bug!("expected capture to be an upvar");
    };
    let HirPlaceBase::Upvar(child_base) = child_capture.place.base else {
        bug!("expected capture to be an upvar");
    };

    parent_base.var_path.hir_id == child_base.var_path.hir_id
        && std::iter::zip(&child_capture.place.projections, &parent_capture.place.projections)
            .all(|(child, parent)| child.kind == parent.kind)
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { closure_typeinfo, ..*providers }
}

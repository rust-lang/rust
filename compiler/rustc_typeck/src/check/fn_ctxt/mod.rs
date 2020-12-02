mod _impl;
mod checks;
mod suggestions;

pub use _impl::*;
pub use checks::*;
pub use suggestions::*;

use crate::astconv::AstConv;
use crate::check::coercion::DynamicCoerceMany;
use crate::check::{Diverges, EnclosingBreakables, Inherited, UnsafetyState};

use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::hir::map::blocks::FnLikeNode;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::{self, Const, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::{self, Span};
use rustc_trait_selection::traits::{ObligationCause, ObligationCauseCode};

use std::cell::{Cell, RefCell};
use std::ops::Deref;

pub struct FnCtxt<'a, 'tcx> {
    pub(super) body_id: hir::HirId,

    /// The parameter environment used for proving trait obligations
    /// in this function. This can change when we descend into
    /// closures (as they bring new things into scope), hence it is
    /// not part of `Inherited` (as of the time of this writing,
    /// closures do not yet change the environment, but they will
    /// eventually).
    pub(super) param_env: ty::ParamEnv<'tcx>,

    /// Number of errors that had been reported when we started
    /// checking this function. On exit, if we find that *more* errors
    /// have been reported, we will skip regionck and other work that
    /// expects the types within the function to be consistent.
    // FIXME(matthewjasper) This should not exist, and it's not correct
    // if type checking is run in parallel.
    err_count_on_creation: usize,

    /// If `Some`, this stores coercion information for returned
    /// expressions. If `None`, this is in a context where return is
    /// inappropriate, such as a const expression.
    ///
    /// This is a `RefCell<DynamicCoerceMany>`, which means that we
    /// can track all the return expressions and then use them to
    /// compute a useful coercion from the set, similar to a match
    /// expression or other branching context. You can use methods
    /// like `expected_ty` to access the declared return type (if
    /// any).
    pub(super) ret_coercion: Option<RefCell<DynamicCoerceMany<'tcx>>>,

    pub(super) ret_coercion_impl_trait: Option<Ty<'tcx>>,

    pub(super) ret_type_span: Option<Span>,

    /// Used exclusively to reduce cost of advanced evaluation used for
    /// more helpful diagnostics.
    pub(super) in_tail_expr: bool,

    /// First span of a return site that we find. Used in error messages.
    pub(super) ret_coercion_span: RefCell<Option<Span>>,

    pub(super) resume_yield_tys: Option<(Ty<'tcx>, Ty<'tcx>)>,

    pub(super) ps: RefCell<UnsafetyState>,

    /// Whether the last checked node generates a divergence (e.g.,
    /// `return` will set this to `Always`). In general, when entering
    /// an expression or other node in the tree, the initial value
    /// indicates whether prior parts of the containing expression may
    /// have diverged. It is then typically set to `Maybe` (and the
    /// old value remembered) for processing the subparts of the
    /// current expression. As each subpart is processed, they may set
    /// the flag to `Always`, etc. Finally, at the end, we take the
    /// result and "union" it with the original value, so that when we
    /// return the flag indicates if any subpart of the parent
    /// expression (up to and including this part) has diverged. So,
    /// if you read it after evaluating a subexpression `X`, the value
    /// you get indicates whether any subexpression that was
    /// evaluating up to and including `X` diverged.
    ///
    /// We currently use this flag only for diagnostic purposes:
    ///
    /// - To warn about unreachable code: if, after processing a
    ///   sub-expression but before we have applied the effects of the
    ///   current node, we see that the flag is set to `Always`, we
    ///   can issue a warning. This corresponds to something like
    ///   `foo(return)`; we warn on the `foo()` expression. (We then
    ///   update the flag to `WarnedAlways` to suppress duplicate
    ///   reports.) Similarly, if we traverse to a fresh statement (or
    ///   tail expression) from a `Always` setting, we will issue a
    ///   warning. This corresponds to something like `{return;
    ///   foo();}` or `{return; 22}`, where we would warn on the
    ///   `foo()` or `22`.
    ///
    /// An expression represents dead code if, after checking it,
    /// the diverges flag is set to something other than `Maybe`.
    pub(super) diverges: Cell<Diverges>,

    /// Whether any child nodes have any type errors.
    pub(super) has_errors: Cell<bool>,

    pub(super) enclosing_breakables: RefCell<EnclosingBreakables<'tcx>>,

    pub(super) inh: &'a Inherited<'a, 'tcx>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn new(
        inh: &'a Inherited<'a, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: hir::HirId,
    ) -> FnCtxt<'a, 'tcx> {
        FnCtxt {
            body_id,
            param_env,
            err_count_on_creation: inh.tcx.sess.err_count(),
            ret_coercion: None,
            ret_coercion_impl_trait: None,
            ret_type_span: None,
            in_tail_expr: false,
            ret_coercion_span: RefCell::new(None),
            resume_yield_tys: None,
            ps: RefCell::new(UnsafetyState::function(hir::Unsafety::Normal, hir::CRATE_HIR_ID)),
            diverges: Cell::new(Diverges::Maybe),
            has_errors: Cell::new(false),
            enclosing_breakables: RefCell::new(EnclosingBreakables {
                stack: Vec::new(),
                by_id: Default::default(),
            }),
            inh,
        }
    }

    pub fn cause(&self, span: Span, code: ObligationCauseCode<'tcx>) -> ObligationCause<'tcx> {
        ObligationCause::new(span, self.body_id, code)
    }

    pub fn misc(&self, span: Span) -> ObligationCause<'tcx> {
        self.cause(span, ObligationCauseCode::MiscObligation)
    }

    pub fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    pub fn errors_reported_since_creation(&self) -> bool {
        self.tcx.sess.err_count() > self.err_count_on_creation
    }
}

impl<'a, 'tcx> Deref for FnCtxt<'a, 'tcx> {
    type Target = Inherited<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.inh
    }
}

impl<'a, 'tcx> AstConv<'tcx> for FnCtxt<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn item_def_id(&self) -> Option<DefId> {
        None
    }

    fn default_constness_for_trait_bounds(&self) -> hir::Constness {
        // FIXME: refactor this into a method
        let node = self.tcx.hir().get(self.body_id);
        if let Some(fn_like) = FnLikeNode::from_node(node) {
            fn_like.constness()
        } else {
            hir::Constness::NotConst
        }
    }

    fn get_type_parameter_bounds(&self, _: Span, def_id: DefId) -> ty::GenericPredicates<'tcx> {
        let tcx = self.tcx;
        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        let item_id = tcx.hir().ty_param_owner(hir_id);
        let item_def_id = tcx.hir().local_def_id(item_id);
        let generics = tcx.generics_of(item_def_id);
        let index = generics.param_def_id_to_index[&def_id];
        ty::GenericPredicates {
            parent: None,
            predicates: tcx.arena.alloc_from_iter(
                self.param_env.caller_bounds().iter().filter_map(|predicate| {
                    match predicate.skip_binders() {
                        ty::PredicateAtom::Trait(data, _) if data.self_ty().is_param(index) => {
                            // HACK(eddyb) should get the original `Span`.
                            let span = tcx.def_span(def_id);
                            Some((predicate, span))
                        }
                        _ => None,
                    }
                }),
            ),
        }
    }

    fn re_infer(&self, def: Option<&ty::GenericParamDef>, span: Span) -> Option<ty::Region<'tcx>> {
        let v = match def {
            Some(def) => infer::EarlyBoundRegion(span, def.name),
            None => infer::MiscVariable(span),
        };
        Some(self.next_region_var(v))
    }

    fn allow_ty_infer(&self) -> bool {
        true
    }

    fn ty_infer(&self, param: Option<&ty::GenericParamDef>, span: Span) -> Ty<'tcx> {
        if let Some(param) = param {
            if let GenericArgKind::Type(ty) = self.var_for_def(span, param).unpack() {
                return ty;
            }
            unreachable!()
        } else {
            self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span,
            })
        }
    }

    fn ct_infer(
        &self,
        ty: Ty<'tcx>,
        param: Option<&ty::GenericParamDef>,
        span: Span,
    ) -> &'tcx Const<'tcx> {
        if let Some(param) = param {
            if let GenericArgKind::Const(ct) = self.var_for_def(span, param).unpack() {
                return ct;
            }
            unreachable!()
        } else {
            self.next_const_var(
                ty,
                ConstVariableOrigin { kind: ConstVariableOriginKind::ConstInference, span },
            )
        }
    }

    fn projected_ty_from_poly_trait_ref(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx> {
        let (trait_ref, _) = self.replace_bound_vars_with_fresh_vars(
            span,
            infer::LateBoundRegionConversionTime::AssocTypeProjection(item_def_id),
            poly_trait_ref,
        );

        let item_substs = <dyn AstConv<'tcx>>::create_substs_for_associated_item(
            self,
            self.tcx,
            span,
            item_def_id,
            item_segment,
            trait_ref.substs,
        );

        self.tcx().mk_projection(item_def_id, item_substs)
    }

    fn normalize_ty(&self, span: Span, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.has_escaping_bound_vars() {
            ty // FIXME: normalization and escaping regions
        } else {
            self.normalize_associated_types_in(span, &ty)
        }
    }

    fn set_tainted_by_errors(&self) {
        self.infcx.set_tainted_by_errors()
    }

    fn record_ty(&self, hir_id: hir::HirId, ty: Ty<'tcx>, _span: Span) {
        self.write_ty(hir_id, ty)
    }
}

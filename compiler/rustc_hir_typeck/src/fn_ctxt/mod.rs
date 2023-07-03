mod _impl;
mod adjust_fulfillment_errors;
mod arg_matrix;
mod checks;
mod suggestions;

pub use _impl::*;
use rustc_errors::ErrorGuaranteed;
pub use suggestions::*;

use crate::coercion::DynamicCoerceMany;
use crate::{Diverges, EnclosingBreakables, Inherited};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir_analysis::astconv::AstConv;
use rustc_infer::infer;
use rustc_infer::infer::error_reporting::TypeErrCtxt;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::ty::{self, Const, Ty, TyCtxt, TypeVisitableExt};
use rustc_session::Session;
use rustc_span::symbol::Ident;
use rustc_span::{self, Span, DUMMY_SP};
use rustc_trait_selection::traits::{ObligationCause, ObligationCauseCode, ObligationCtxt};

use std::cell::{Cell, RefCell};
use std::ops::Deref;

/// The `FnCtxt` stores type-checking context needed to type-check bodies of
/// functions, closures, and `const`s, including performing type inference
/// with [`InferCtxt`].
///
/// This is in contrast to [`ItemCtxt`], which is used to type-check item *signatures*
/// and thus does not perform type inference.
///
/// See [`ItemCtxt`]'s docs for more.
///
/// [`ItemCtxt`]: rustc_hir_analysis::collect::ItemCtxt
/// [`InferCtxt`]: infer::InferCtxt
pub struct FnCtxt<'a, 'tcx> {
    pub(super) body_id: LocalDefId,

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

    /// First span of a return site that we find. Used in error messages.
    pub(super) ret_coercion_span: Cell<Option<Span>>,

    pub(super) resume_yield_tys: Option<(Ty<'tcx>, Ty<'tcx>)>,

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
    ///   tail expression) from an `Always` setting, we will issue a
    ///   warning. This corresponds to something like `{return;
    ///   foo();}` or `{return; 22}`, where we would warn on the
    ///   `foo()` or `22`.
    ///
    /// An expression represents dead code if, after checking it,
    /// the diverges flag is set to something other than `Maybe`.
    pub(super) diverges: Cell<Diverges>,

    pub(super) enclosing_breakables: RefCell<EnclosingBreakables<'tcx>>,

    pub(super) inh: &'a Inherited<'tcx>,

    pub(super) fallback_has_occurred: Cell<bool>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn new(
        inh: &'a Inherited<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body_id: LocalDefId,
    ) -> FnCtxt<'a, 'tcx> {
        FnCtxt {
            body_id,
            param_env,
            err_count_on_creation: inh.tcx.sess.err_count(),
            ret_coercion: None,
            ret_coercion_span: Cell::new(None),
            resume_yield_tys: None,
            diverges: Cell::new(Diverges::Maybe),
            enclosing_breakables: RefCell::new(EnclosingBreakables {
                stack: Vec::new(),
                by_id: Default::default(),
            }),
            inh,
            fallback_has_occurred: Cell::new(false),
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

    /// Creates an `TypeErrCtxt` with a reference to the in-progress
    /// `TypeckResults` which is used for diagnostics.
    /// Use [`InferCtxt::err_ctxt`] to start one without a `TypeckResults`.
    ///
    /// [`InferCtxt::err_ctxt`]: infer::InferCtxt::err_ctxt
    pub fn err_ctxt(&'a self) -> TypeErrCtxt<'a, 'tcx> {
        TypeErrCtxt {
            infcx: &self.infcx,
            typeck_results: Some(self.typeck_results.borrow()),
            fallback_has_occurred: self.fallback_has_occurred.get(),
            normalize_fn_sig: Box::new(|fn_sig| {
                if fn_sig.has_escaping_bound_vars() {
                    return fn_sig;
                }
                self.probe(|_| {
                    let ocx = ObligationCtxt::new(self);
                    let normalized_fn_sig =
                        ocx.normalize(&ObligationCause::dummy(), self.param_env, fn_sig);
                    if ocx.select_all_or_error().is_empty() {
                        let normalized_fn_sig = self.resolve_vars_if_possible(normalized_fn_sig);
                        if !normalized_fn_sig.has_infer() {
                            return normalized_fn_sig;
                        }
                    }
                    fn_sig
                })
            }),
            autoderef_steps: Box::new(|ty| {
                let mut autoderef = self.autoderef(DUMMY_SP, ty).silence_errors();
                let mut steps = vec![];
                while let Some((ty, _)) = autoderef.next() {
                    steps.push((ty, autoderef.current_obligations()));
                }
                steps
            }),
        }
    }

    pub fn errors_reported_since_creation(&self) -> bool {
        self.tcx.sess.err_count() > self.err_count_on_creation
    }

    pub fn next_root_ty_var(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        self.tcx.mk_ty_var(self.next_ty_var_id_in_universe(origin, ty::UniverseIndex::ROOT))
    }
}

impl<'a, 'tcx> Deref for FnCtxt<'a, 'tcx> {
    type Target = Inherited<'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.inh
    }
}

impl<'a, 'tcx> AstConv<'tcx> for FnCtxt<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn item_def_id(&self) -> DefId {
        self.body_id.to_def_id()
    }

    fn get_type_parameter_bounds(
        &self,
        _: Span,
        def_id: LocalDefId,
        _: Ident,
    ) -> ty::GenericPredicates<'tcx> {
        let tcx = self.tcx;
        let item_def_id = tcx.hir().ty_param_owner(def_id);
        let generics = tcx.generics_of(item_def_id);
        let index = generics.param_def_id_to_index[&def_id.to_def_id()];
        ty::GenericPredicates {
            parent: None,
            predicates: tcx.arena.alloc_from_iter(
                self.param_env.caller_bounds().iter().filter_map(|predicate| {
                    match predicate.kind().skip_binder() {
                        ty::ClauseKind::Trait(data) if data.self_ty().is_param(index) => {
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
        match param {
            Some(param) => self.var_for_def(span, param).as_type().unwrap(),
            None => self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::TypeInference,
                span,
            }),
        }
    }

    fn ct_infer(
        &self,
        ty: Ty<'tcx>,
        param: Option<&ty::GenericParamDef>,
        span: Span,
    ) -> Const<'tcx> {
        match param {
            Some(param) => self.var_for_def(span, param).as_const().unwrap(),
            None => self.next_const_var(
                ty,
                ConstVariableOrigin { kind: ConstVariableOriginKind::ConstInference, span },
            ),
        }
    }

    fn projected_ty_from_poly_trait_ref(
        &self,
        span: Span,
        item_def_id: DefId,
        item_segment: &hir::PathSegment<'_>,
        poly_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Ty<'tcx> {
        let trait_ref = self.instantiate_binder_with_fresh_vars(
            span,
            infer::LateBoundRegionConversionTime::AssocTypeProjection(item_def_id),
            poly_trait_ref,
        );

        let item_substs = self.astconv().create_substs_for_associated_item(
            span,
            item_def_id,
            item_segment,
            trait_ref.substs,
        );

        self.tcx().mk_projection(item_def_id, item_substs)
    }

    fn probe_adt(&self, span: Span, ty: Ty<'tcx>) -> Option<ty::AdtDef<'tcx>> {
        match ty.kind() {
            ty::Adt(adt_def, _) => Some(*adt_def),
            // FIXME(#104767): Should we handle bound regions here?
            ty::Alias(ty::Projection | ty::Inherent, _) if !ty.has_escaping_bound_vars() => {
                self.normalize(span, ty).ty_adt_def()
            }
            _ => None,
        }
    }

    fn set_tainted_by_errors(&self, e: ErrorGuaranteed) {
        self.infcx.set_tainted_by_errors(e)
    }

    fn record_ty(&self, hir_id: hir::HirId, ty: Ty<'tcx>, span: Span) {
        // FIXME: normalization and escaping regions
        let ty = if !ty.has_escaping_bound_vars() { self.normalize(span, ty) } else { ty };
        self.write_ty(hir_id, ty)
    }

    fn infcx(&self) -> Option<&infer::InferCtxt<'tcx>> {
        Some(&self.infcx)
    }
}

/// Represents a user-provided type in the raw form (never normalized).
///
/// This is a bridge between the interface of `AstConv`, which outputs a raw `Ty`,
/// and the API in this module, which expect `Ty` to be fully normalized.
#[derive(Clone, Copy, Debug)]
pub struct RawTy<'tcx> {
    pub raw: Ty<'tcx>,

    /// The normalized form of `raw`, stored here for efficiency.
    pub normalized: Ty<'tcx>,
}

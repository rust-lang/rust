//! Method lookup: the secret sauce of Rust. See the [rustc dev guide] for more information.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/method-lookup.html

mod confirm;
mod prelude_edition_lints;
pub(crate) mod probe;
mod suggest;

use rustc_errors::{Applicability, Diag, SubdiagMessage};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::{self, InferOk};
use rustc_infer::traits::PredicateObligations;
use rustc_middle::query::Providers;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{
    self, GenericArgs, GenericArgsRef, GenericParamDefKind, Ty, TypeVisitableExt,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{ErrorGuaranteed, Ident, Span};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{self, NormalizeExt};
use tracing::{debug, instrument};

pub(crate) use self::MethodError::*;
use self::probe::{IsSuggestion, ProbeScope};
use crate::FnCtxt;

pub(crate) fn provide(providers: &mut Providers) {
    probe::provide(providers);
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MethodCallee<'tcx> {
    /// Impl method ID, for inherent methods, or trait method ID, otherwise.
    pub def_id: DefId,
    pub args: GenericArgsRef<'tcx>,

    /// Instantiated method signature, i.e., it has been
    /// instantiated, normalized, and has had late-bound
    /// lifetimes replaced with inference variables.
    pub sig: ty::FnSig<'tcx>,
}

#[derive(Debug)]
pub(crate) enum MethodError<'tcx> {
    /// Did not find an applicable method, but we did find various near-misses that may work.
    NoMatch(NoMatchData<'tcx>),

    /// Multiple methods might apply.
    Ambiguity(Vec<CandidateSource>),

    /// Found an applicable method, but it is not visible. The third argument contains a list of
    /// not-in-scope traits which may work.
    PrivateMatch(DefKind, DefId, Vec<DefId>),

    /// Found a `Self: Sized` bound where `Self` is a trait object.
    IllegalSizedBound {
        candidates: Vec<DefId>,
        needs_mut: bool,
        bound_span: Span,
        self_expr: &'tcx hir::Expr<'tcx>,
    },

    /// Found a match, but the return type is wrong
    BadReturnType,

    /// Error has already been emitted, no need to emit another one.
    ErrorReported(ErrorGuaranteed),
}

// Contains a list of static methods that may apply, a list of unsatisfied trait predicates which
// could lead to matches if satisfied, and a list of not-in-scope traits which may work.
#[derive(Debug)]
pub(crate) struct NoMatchData<'tcx> {
    pub static_candidates: Vec<CandidateSource>,
    pub unsatisfied_predicates:
        Vec<(ty::Predicate<'tcx>, Option<ty::Predicate<'tcx>>, Option<ObligationCause<'tcx>>)>,
    pub out_of_scope_traits: Vec<DefId>,
    pub similar_candidate: Option<ty::AssocItem>,
    pub mode: probe::Mode,
}

// A pared down enum describing just the places from which a method
// candidate can arise. Used for error reporting only.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum CandidateSource {
    Impl(DefId),
    Trait(DefId /* trait id */),
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Determines whether the type `self_ty` supports a visible method named `method_name` or not.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn method_exists_for_diagnostic(
        &self,
        method_name: Ident,
        self_ty: Ty<'tcx>,
        call_expr_id: hir::HirId,
        return_type: Option<Ty<'tcx>>,
    ) -> bool {
        match self.probe_for_name(
            probe::Mode::MethodCall,
            method_name,
            return_type,
            IsSuggestion(true),
            self_ty,
            call_expr_id,
            ProbeScope::TraitsInScope,
        ) {
            Ok(pick) => {
                pick.maybe_emit_unstable_name_collision_hint(
                    self.tcx,
                    method_name.span,
                    call_expr_id,
                );
                true
            }
            Err(NoMatch(..)) => false,
            Err(Ambiguity(..)) => true,
            Err(PrivateMatch(..)) => false,
            Err(IllegalSizedBound { .. }) => true,
            Err(BadReturnType) => false,
            Err(ErrorReported(_)) => false,
        }
    }

    /// Adds a suggestion to call the given method to the provided diagnostic.
    #[instrument(level = "debug", skip(self, err, call_expr))]
    pub(crate) fn suggest_method_call(
        &self,
        err: &mut Diag<'_>,
        msg: impl Into<SubdiagMessage> + std::fmt::Debug,
        method_name: Ident,
        self_ty: Ty<'tcx>,
        call_expr: &hir::Expr<'tcx>,
        span: Option<Span>,
    ) {
        let params = self
            .lookup_probe_for_diagnostic(
                method_name,
                self_ty,
                call_expr,
                ProbeScope::TraitsInScope,
                None,
            )
            .map(|pick| {
                let sig = self.tcx.fn_sig(pick.item.def_id);
                sig.skip_binder().inputs().skip_binder().len().saturating_sub(1)
            })
            .unwrap_or(0);

        // Account for `foo.bar<T>`;
        let sugg_span = span.unwrap_or(call_expr.span).shrink_to_hi();
        let (suggestion, applicability) = (
            format!("({})", (0..params).map(|_| "_").collect::<Vec<_>>().join(", ")),
            if params > 0 { Applicability::HasPlaceholders } else { Applicability::MaybeIncorrect },
        );

        err.span_suggestion_verbose(sugg_span, msg, suggestion, applicability);
    }

    /// Performs method lookup. If lookup is successful, it will return the callee
    /// and store an appropriate adjustment for the self-expr. In some cases it may
    /// report an error (e.g., invoking the `drop` method).
    ///
    /// # Arguments
    ///
    /// Given a method call like `foo.bar::<T1,...Tn>(a, b + 1, ...)`:
    ///
    /// * `self`:                  the surrounding `FnCtxt` (!)
    /// * `self_ty`:               the (unadjusted) type of the self expression (`foo`)
    /// * `segment`:               the name and generic arguments of the method (`bar::<T1, ...Tn>`)
    /// * `span`:                  the span for the method call
    /// * `call_expr`:             the complete method call: (`foo.bar::<T1,...Tn>(...)`)
    /// * `self_expr`:             the self expression (`foo`)
    /// * `args`:                  the expressions of the arguments (`a, b + 1, ...`)
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn lookup_method(
        &self,
        self_ty: Ty<'tcx>,
        segment: &'tcx hir::PathSegment<'tcx>,
        span: Span,
        call_expr: &'tcx hir::Expr<'tcx>,
        self_expr: &'tcx hir::Expr<'tcx>,
        args: &'tcx [hir::Expr<'tcx>],
    ) -> Result<MethodCallee<'tcx>, MethodError<'tcx>> {
        let scope = if let Some(only_method) = segment.res.opt_def_id() {
            ProbeScope::Single(only_method)
        } else {
            ProbeScope::TraitsInScope
        };

        let pick = self.lookup_probe(segment.ident, self_ty, call_expr, scope)?;

        self.lint_edition_dependent_dot_call(
            self_ty, segment, span, call_expr, self_expr, &pick, args,
        );

        // NOTE: on the failure path, we also record the possibly-used trait methods
        // since an unused import warning is kinda distracting from the method error.
        for &import_id in &pick.import_ids {
            debug!("used_trait_import: {:?}", import_id);
            self.typeck_results.borrow_mut().used_trait_imports.insert(import_id);
        }

        self.tcx.check_stability(pick.item.def_id, Some(call_expr.hir_id), span, None);

        let result = self.confirm_method(span, self_expr, call_expr, self_ty, &pick, segment);
        debug!("result = {:?}", result);

        if let Some(span) = result.illegal_sized_bound {
            let mut needs_mut = false;
            if let ty::Ref(region, t_type, mutability) = self_ty.kind() {
                let trait_type = Ty::new_ref(self.tcx, *region, *t_type, mutability.invert());
                // We probe again to see if there might be a borrow mutability discrepancy.
                match self.lookup_probe(
                    segment.ident,
                    trait_type,
                    call_expr,
                    ProbeScope::TraitsInScope,
                ) {
                    Ok(ref new_pick) if pick.differs_from(new_pick) => {
                        needs_mut = new_pick.self_ty.ref_mutability() != self_ty.ref_mutability();
                    }
                    _ => {}
                }
            }

            // We probe again, taking all traits into account (not only those in scope).
            let candidates = match self.lookup_probe_for_diagnostic(
                segment.ident,
                self_ty,
                call_expr,
                ProbeScope::AllTraits,
                None,
            ) {
                // If we find a different result the caller probably forgot to import a trait.
                Ok(ref new_pick) if pick.differs_from(new_pick) => {
                    vec![new_pick.item.container_id(self.tcx)]
                }
                Err(Ambiguity(ref sources)) => sources
                    .iter()
                    .filter_map(|source| {
                        match *source {
                            // Note: this cannot come from an inherent impl,
                            // because the first probing succeeded.
                            CandidateSource::Impl(def) => self.tcx.trait_id_of_impl(def),
                            CandidateSource::Trait(_) => None,
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };

            return Err(IllegalSizedBound { candidates, needs_mut, bound_span: span, self_expr });
        }

        Ok(result.callee)
    }

    pub(crate) fn lookup_method_for_diagnostic(
        &self,
        self_ty: Ty<'tcx>,
        segment: &hir::PathSegment<'tcx>,
        span: Span,
        call_expr: &'tcx hir::Expr<'tcx>,
        self_expr: &'tcx hir::Expr<'tcx>,
    ) -> Result<MethodCallee<'tcx>, MethodError<'tcx>> {
        let pick = self.lookup_probe_for_diagnostic(
            segment.ident,
            self_ty,
            call_expr,
            ProbeScope::TraitsInScope,
            None,
        )?;

        Ok(self
            .confirm_method_for_diagnostic(span, self_expr, call_expr, self_ty, &pick, segment)
            .callee)
    }

    #[instrument(level = "debug", skip(self, call_expr))]
    pub(crate) fn lookup_probe(
        &self,
        method_name: Ident,
        self_ty: Ty<'tcx>,
        call_expr: &hir::Expr<'_>,
        scope: ProbeScope,
    ) -> probe::PickResult<'tcx> {
        let pick = self.probe_for_name(
            probe::Mode::MethodCall,
            method_name,
            None,
            IsSuggestion(false),
            self_ty,
            call_expr.hir_id,
            scope,
        )?;
        pick.maybe_emit_unstable_name_collision_hint(self.tcx, method_name.span, call_expr.hir_id);
        Ok(pick)
    }

    pub(crate) fn lookup_probe_for_diagnostic(
        &self,
        method_name: Ident,
        self_ty: Ty<'tcx>,
        call_expr: &hir::Expr<'_>,
        scope: ProbeScope,
        return_type: Option<Ty<'tcx>>,
    ) -> probe::PickResult<'tcx> {
        let pick = self.probe_for_name(
            probe::Mode::MethodCall,
            method_name,
            return_type,
            IsSuggestion(true),
            self_ty,
            call_expr.hir_id,
            scope,
        )?;
        Ok(pick)
    }

    /// `lookup_method_in_trait` is used for overloaded operators.
    /// It does a very narrow slice of what the normal probe/confirm path does.
    /// In particular, it doesn't really do any probing: it simply constructs
    /// an obligation for a particular trait with the given self type and checks
    /// whether that trait is implemented.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn lookup_method_in_trait(
        &self,
        cause: ObligationCause<'tcx>,
        m_name: Ident,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        opt_rhs_ty: Option<Ty<'tcx>>,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        // Construct a trait-reference `self_ty : Trait<input_tys>`
        let args = GenericArgs::for_item(self.tcx, trait_def_id, |param, _| match param.kind {
            GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => {
                unreachable!("did not expect operator trait to have lifetime/const")
            }
            GenericParamDefKind::Type { .. } => {
                if param.index == 0 {
                    self_ty.into()
                } else if let Some(rhs_ty) = opt_rhs_ty {
                    assert_eq!(param.index, 1, "did not expect >1 param on operator trait");
                    rhs_ty.into()
                } else {
                    // FIXME: We should stop passing `None` for the failure case
                    // when probing for call exprs. I.e. `opt_rhs_ty` should always
                    // be set when it needs to be.
                    self.var_for_def(cause.span, param)
                }
            }
        });

        let obligation = traits::Obligation::new(
            self.tcx,
            cause,
            self.param_env,
            ty::TraitRef::new_from_args(self.tcx, trait_def_id, args),
        );

        // Now we want to know if this can be matched
        if !self.predicate_may_hold(&obligation) {
            debug!("--> Cannot match obligation");
            // Cannot be matched, no such method resolution is possible.
            return None;
        }

        // Trait must have a method named `m_name` and it should not have
        // type parameters or early-bound regions.
        let tcx = self.tcx;
        let Some(method_item) = self.associated_value(trait_def_id, m_name) else {
            bug!("expected associated item for operator trait")
        };

        let def_id = method_item.def_id;
        if !method_item.is_fn() {
            span_bug!(tcx.def_span(def_id), "expected `{m_name}` to be an associated function");
        }

        debug!("lookup_in_trait_adjusted: method_item={:?}", method_item);
        let mut obligations = PredicateObligations::new();

        // Instantiate late-bound regions and instantiate the trait
        // parameters into the method type to get the actual method type.
        //
        // N.B., instantiate late-bound regions before normalizing the
        // function signature so that normalization does not need to deal
        // with bound regions.
        let fn_sig = tcx.fn_sig(def_id).instantiate(self.tcx, args);
        let fn_sig =
            self.instantiate_binder_with_fresh_vars(obligation.cause.span, infer::FnCall, fn_sig);

        let InferOk { value: fn_sig, obligations: o } =
            self.at(&obligation.cause, self.param_env).normalize(fn_sig);
        obligations.extend(o);

        // Register obligations for the parameters. This will include the
        // `Self` parameter, which in turn has a bound of the main trait,
        // so this also effectively registers `obligation` as well. (We
        // used to register `obligation` explicitly, but that resulted in
        // double error messages being reported.)
        //
        // Note that as the method comes from a trait, it should not have
        // any late-bound regions appearing in its bounds.
        let bounds = self.tcx.predicates_of(def_id).instantiate(self.tcx, args);

        let InferOk { value: bounds, obligations: o } =
            self.at(&obligation.cause, self.param_env).normalize(bounds);
        obligations.extend(o);
        assert!(!bounds.has_escaping_bound_vars());

        let predicates_cause = obligation.cause.clone();
        obligations.extend(traits::predicates_for_generics(
            move |_, _| predicates_cause.clone(),
            self.param_env,
            bounds,
        ));

        // Also add an obligation for the method type being well-formed.
        let method_ty = Ty::new_fn_ptr(tcx, ty::Binder::dummy(fn_sig));
        debug!(
            "lookup_method_in_trait: matched method method_ty={:?} obligation={:?}",
            method_ty, obligation
        );
        obligations.push(traits::Obligation::new(
            tcx,
            obligation.cause,
            self.param_env,
            ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                method_ty.into(),
            ))),
        ));

        let callee = MethodCallee { def_id, args, sig: fn_sig };
        debug!("callee = {:?}", callee);

        Some(InferOk { obligations, value: callee })
    }

    /// Performs a [full-qualified function call] (formerly "universal function call") lookup. If
    /// lookup is successful, it will return the type of definition and the [`DefId`] of the found
    /// function definition.
    ///
    /// [full-qualified function call]: https://doc.rust-lang.org/reference/expressions/call-expr.html#disambiguating-function-calls
    ///
    /// # Arguments
    ///
    /// Given a function call like `Foo::bar::<T1,...Tn>(...)`:
    ///
    /// * `self`:                  the surrounding `FnCtxt` (!)
    /// * `span`:                  the span of the call, excluding arguments (`Foo::bar::<T1, ...Tn>`)
    /// * `method_name`:           the identifier of the function within the container type (`bar`)
    /// * `self_ty`:               the type to search within (`Foo`)
    /// * `self_ty_span`           the span for the type being searched within (span of `Foo`)
    /// * `expr_id`:               the [`hir::HirId`] of the expression composing the entire call
    #[instrument(level = "debug", skip(self), ret)]
    pub(crate) fn resolve_fully_qualified_call(
        &self,
        span: Span,
        method_name: Ident,
        self_ty: Ty<'tcx>,
        self_ty_span: Span,
        expr_id: hir::HirId,
    ) -> Result<(DefKind, DefId), MethodError<'tcx>> {
        let tcx = self.tcx;

        // Check if we have an enum variant.
        let mut struct_variant = None;
        if let ty::Adt(adt_def, _) = self_ty.kind() {
            if adt_def.is_enum() {
                let variant_def = adt_def
                    .variants()
                    .iter()
                    .find(|vd| tcx.hygienic_eq(method_name, vd.ident(tcx), adt_def.did()));
                if let Some(variant_def) = variant_def {
                    if let Some((ctor_kind, ctor_def_id)) = variant_def.ctor {
                        tcx.check_stability(
                            ctor_def_id,
                            Some(expr_id),
                            span,
                            Some(method_name.span),
                        );
                        return Ok((DefKind::Ctor(CtorOf::Variant, ctor_kind), ctor_def_id));
                    } else {
                        struct_variant = Some((DefKind::Variant, variant_def.def_id));
                    }
                }
            }
        }

        let pick = self.probe_for_name(
            probe::Mode::Path,
            method_name,
            None,
            IsSuggestion(false),
            self_ty,
            expr_id,
            ProbeScope::TraitsInScope,
        );
        let pick = match (pick, struct_variant) {
            // Fall back to a resolution that will produce an error later.
            (Err(_), Some(res)) => return Ok(res),
            (pick, _) => pick?,
        };

        pick.maybe_emit_unstable_name_collision_hint(self.tcx, span, expr_id);

        self.lint_fully_qualified_call_from_2018(
            span,
            method_name,
            self_ty,
            self_ty_span,
            expr_id,
            &pick,
        );

        debug!(?pick);
        {
            let mut typeck_results = self.typeck_results.borrow_mut();
            for import_id in pick.import_ids {
                debug!(used_trait_import=?import_id);
                typeck_results.used_trait_imports.insert(import_id);
            }
        }

        let def_kind = pick.item.as_def_kind();
        tcx.check_stability(pick.item.def_id, Some(expr_id), span, Some(method_name.span));
        Ok((def_kind, pick.item.def_id))
    }

    /// Finds item with name `item_ident` defined in impl/trait `def_id`
    /// and return it, or `None`, if no such item was defined there.
    fn associated_value(&self, def_id: DefId, item_ident: Ident) -> Option<ty::AssocItem> {
        self.tcx
            .associated_items(def_id)
            .find_by_ident_and_namespace(self.tcx, item_ident, Namespace::ValueNS, def_id)
            .copied()
    }
}

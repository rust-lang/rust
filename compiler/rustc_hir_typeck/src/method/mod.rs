//! Method lookup: the secret sauce of Rust. See the [rustc dev guide] for more information.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/method-lookup.html

mod confirm;
mod prelude2021;
pub mod probe;
mod suggest;

pub use self::suggest::SelfSource;
pub use self::MethodError::*;

use crate::errors::OpMethodGenericParams;
use crate::FnCtxt;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Namespace};
use rustc_hir::def_id::DefId;
use rustc_infer::infer::{self, InferOk};
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::subst::{InternalSubsts, SubstsRef};
use rustc_middle::ty::{self, GenericParamDefKind, Ty, TypeVisitable};
use rustc_span::symbol::Ident;
use rustc_span::Span;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{self, NormalizeExt};

use self::probe::{IsSuggestion, ProbeScope};

pub fn provide(providers: &mut ty::query::Providers) {
    probe::provide(providers);
}

#[derive(Clone, Copy, Debug)]
pub struct MethodCallee<'tcx> {
    /// Impl method ID, for inherent methods, or trait method ID, otherwise.
    pub def_id: DefId,
    pub substs: SubstsRef<'tcx>,

    /// Instantiated method signature, i.e., it has been
    /// substituted, normalized, and has had late-bound
    /// lifetimes replaced with inference variables.
    pub sig: ty::FnSig<'tcx>,
}

#[derive(Debug)]
pub enum MethodError<'tcx> {
    // Did not find an applicable method, but we did find various near-misses that may work.
    NoMatch(NoMatchData<'tcx>),

    // Multiple methods might apply.
    Ambiguity(Vec<CandidateSource>),

    // Found an applicable method, but it is not visible. The third argument contains a list of
    // not-in-scope traits which may work.
    PrivateMatch(DefKind, DefId, Vec<DefId>),

    // Found a `Self: Sized` bound where `Self` is a trait object.
    IllegalSizedBound {
        candidates: Vec<DefId>,
        needs_mut: bool,
        bound_span: Span,
        self_expr: &'tcx hir::Expr<'tcx>,
    },

    // Found a match, but the return type is wrong
    BadReturnType,
}

// Contains a list of static methods that may apply, a list of unsatisfied trait predicates which
// could lead to matches if satisfied, and a list of not-in-scope traits which may work.
#[derive(Debug)]
pub struct NoMatchData<'tcx> {
    pub static_candidates: Vec<CandidateSource>,
    pub unsatisfied_predicates:
        Vec<(ty::Predicate<'tcx>, Option<ty::Predicate<'tcx>>, Option<ObligationCause<'tcx>>)>,
    pub out_of_scope_traits: Vec<DefId>,
    pub similar_candidate: Option<ty::AssocItem>,
    pub mode: probe::Mode,
}

// A pared down enum describing just the places from which a method
// candidate can arise. Used for error reporting only.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CandidateSource {
    Impl(DefId),
    Trait(DefId /* trait id */),
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Determines whether the type `self_ty` supports a method name `method_name` or not.
    #[instrument(level = "debug", skip(self))]
    pub fn method_exists(
        &self,
        method_name: Ident,
        self_ty: Ty<'tcx>,
        call_expr_id: hir::HirId,
        allow_private: bool,
        return_type: Option<Ty<'tcx>>,
    ) -> bool {
        match self.probe_for_name(
            probe::Mode::MethodCall,
            method_name,
            return_type,
            IsSuggestion(false),
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
            Err(PrivateMatch(..)) => allow_private,
            Err(IllegalSizedBound { .. }) => true,
            Err(BadReturnType) => false,
        }
    }

    /// Adds a suggestion to call the given method to the provided diagnostic.
    #[instrument(level = "debug", skip(self, err, call_expr))]
    pub(crate) fn suggest_method_call(
        &self,
        err: &mut Diagnostic,
        msg: &str,
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
    pub fn lookup_method(
        &self,
        self_ty: Ty<'tcx>,
        segment: &hir::PathSegment<'_>,
        span: Span,
        call_expr: &'tcx hir::Expr<'tcx>,
        self_expr: &'tcx hir::Expr<'tcx>,
        args: &'tcx [hir::Expr<'tcx>],
    ) -> Result<MethodCallee<'tcx>, MethodError<'tcx>> {
        let pick =
            self.lookup_probe(segment.ident, self_ty, call_expr, ProbeScope::TraitsInScope)?;

        self.lint_dot_call_from_2018(self_ty, segment, span, call_expr, self_expr, &pick, args);

        for import_id in &pick.import_ids {
            debug!("used_trait_import: {:?}", import_id);
            Lrc::get_mut(&mut self.typeck_results.borrow_mut().used_trait_imports)
                .unwrap()
                .insert(*import_id);
        }

        self.tcx.check_stability(pick.item.def_id, Some(call_expr.hir_id), span, None);

        let result = self.confirm_method(span, self_expr, call_expr, self_ty, &pick, segment);
        debug!("result = {:?}", result);

        if let Some(span) = result.illegal_sized_bound {
            let mut needs_mut = false;
            if let ty::Ref(region, t_type, mutability) = self_ty.kind() {
                let trait_type = self
                    .tcx
                    .mk_ref(*region, ty::TypeAndMut { ty: *t_type, mutbl: mutability.invert() });
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

    #[instrument(level = "debug", skip(self, call_expr))]
    pub fn lookup_probe(
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

    pub fn lookup_probe_for_diagnostic(
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

    pub(super) fn obligation_for_method(
        &self,
        cause: ObligationCause<'tcx>,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        opt_input_types: Option<&[Ty<'tcx>]>,
    ) -> (traits::Obligation<'tcx, ty::Predicate<'tcx>>, &'tcx ty::List<ty::subst::GenericArg<'tcx>>)
    {
        // Construct a trait-reference `self_ty : Trait<input_tys>`
        let substs = InternalSubsts::for_item(self.tcx, trait_def_id, |param, _| {
            match param.kind {
                GenericParamDefKind::Lifetime | GenericParamDefKind::Const { .. } => {}
                GenericParamDefKind::Type { .. } => {
                    if param.index == 0 {
                        return self_ty.into();
                    } else if let Some(input_types) = opt_input_types {
                        return input_types[param.index as usize - 1].into();
                    }
                }
            }
            self.var_for_def(cause.span, param)
        });

        let trait_ref = self.tcx.mk_trait_ref(trait_def_id, substs);

        // Construct an obligation
        let poly_trait_ref = ty::Binder::dummy(trait_ref);
        (
            traits::Obligation::new(
                self.tcx,
                cause,
                self.param_env,
                poly_trait_ref.without_const(),
            ),
            substs,
        )
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
        opt_input_types: Option<&[Ty<'tcx>]>,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        let (obligation, substs) =
            self.obligation_for_method(cause, trait_def_id, self_ty, opt_input_types);
        self.construct_obligation_for_trait(m_name, trait_def_id, obligation, substs)
    }

    // FIXME(#18741): it seems likely that we can consolidate some of this
    // code with the other method-lookup code. In particular, the second half
    // of this method is basically the same as confirmation.
    fn construct_obligation_for_trait(
        &self,
        m_name: Ident,
        trait_def_id: DefId,
        obligation: traits::PredicateObligation<'tcx>,
        substs: &'tcx ty::List<ty::subst::GenericArg<'tcx>>,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        debug!(?obligation);

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
            tcx.sess.delay_span_bug(
                obligation.cause.span,
                "operator trait does not have corresponding operator method",
            );
            return None;
        };

        if method_item.kind != ty::AssocKind::Fn {
            self.tcx.sess.delay_span_bug(tcx.def_span(method_item.def_id), "not a method");
            return None;
        }

        let def_id = method_item.def_id;
        let generics = tcx.generics_of(def_id);

        if generics.params.len() != 0 {
            tcx.sess.emit_fatal(OpMethodGenericParams {
                span: tcx.def_span(method_item.def_id),
                method_name: m_name.to_string(),
            });
        }

        debug!("lookup_in_trait_adjusted: method_item={:?}", method_item);
        let mut obligations = vec![];

        // Instantiate late-bound regions and substitute the trait
        // parameters into the method type to get the actual method type.
        //
        // N.B., instantiate late-bound regions before normalizing the
        // function signature so that normalization does not need to deal
        // with bound regions.
        let fn_sig = tcx.fn_sig(def_id).subst(self.tcx, substs);
        let fn_sig =
            self.instantiate_binder_with_fresh_vars(obligation.cause.span, infer::FnCall, fn_sig);

        let InferOk { value, obligations: o } =
            self.at(&obligation.cause, self.param_env).normalize(fn_sig);
        let fn_sig = {
            obligations.extend(o);
            value
        };

        // Register obligations for the parameters. This will include the
        // `Self` parameter, which in turn has a bound of the main trait,
        // so this also effectively registers `obligation` as well. (We
        // used to register `obligation` explicitly, but that resulted in
        // double error messages being reported.)
        //
        // Note that as the method comes from a trait, it should not have
        // any late-bound regions appearing in its bounds.
        let bounds = self.tcx.predicates_of(def_id).instantiate(self.tcx, substs);

        let InferOk { value, obligations: o } =
            self.at(&obligation.cause, self.param_env).normalize(bounds);
        let bounds = {
            obligations.extend(o);
            value
        };

        assert!(!bounds.has_escaping_bound_vars());

        let predicates_cause = obligation.cause.clone();
        obligations.extend(traits::predicates_for_generics(
            move |_, _| predicates_cause.clone(),
            self.param_env,
            bounds,
        ));

        // Also add an obligation for the method type being well-formed.
        let method_ty = tcx.mk_fn_ptr(ty::Binder::dummy(fn_sig));
        debug!(
            "lookup_in_trait_adjusted: matched method method_ty={:?} obligation={:?}",
            method_ty, obligation
        );
        obligations.push(traits::Obligation::new(
            tcx,
            obligation.cause,
            self.param_env,
            ty::Binder::dummy(ty::PredicateKind::WellFormed(method_ty.into())),
        ));

        let callee = MethodCallee { def_id, substs, sig: fn_sig };

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
    pub fn resolve_fully_qualified_call(
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
            let used_trait_imports = Lrc::get_mut(&mut typeck_results.used_trait_imports).unwrap();
            for import_id in pick.import_ids {
                debug!(used_trait_import=?import_id);
                used_trait_imports.insert(import_id);
            }
        }

        let def_kind = pick.item.kind.as_def_kind();
        tcx.check_stability(pick.item.def_id, Some(expr_id), span, Some(method_name.span));
        Ok((def_kind, pick.item.def_id))
    }

    /// Finds item with name `item_name` defined in impl/trait `def_id`
    /// and return it, or `None`, if no such item was defined there.
    pub fn associated_value(&self, def_id: DefId, item_name: Ident) -> Option<ty::AssocItem> {
        self.tcx
            .associated_items(def_id)
            .find_by_name_and_namespace(self.tcx, item_name, Namespace::ValueNS, def_id)
            .copied()
    }
}

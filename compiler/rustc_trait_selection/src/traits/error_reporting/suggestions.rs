use super::{
    EvaluationResult, Obligation, ObligationCause, ObligationCauseCode, PredicateObligation,
    SelectionContext,
};

use crate::autoderef::Autoderef;
use crate::infer::InferCtxt;
use crate::traits::normalize_projection_type;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{error_code, struct_span_err, Applicability, DiagnosticBuilder, Style};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{AsyncGeneratorKind, GeneratorKind, Node};
use rustc_middle::ty::{
    self, suggest_constraining_type_param, AdtKind, DefIdTree, Infer, InferTy, ToPredicate, Ty,
    TyCtxt, TypeFoldable, WithConstness,
};
use rustc_middle::ty::{TypeAndMut, TypeckResults};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, MultiSpan, Span, DUMMY_SP};
use rustc_target::spec::abi;
use std::fmt;

use super::InferCtxtPrivExt;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;

#[derive(Debug)]
pub enum GeneratorInteriorOrUpvar {
    // span of interior type
    Interior(Span),
    // span of upvar
    Upvar(Span),
}

// This trait is public to expose the diagnostics methods to clippy.
pub trait InferCtxtExt<'tcx> {
    fn suggest_restricting_param_bound(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        body_id: hir::HirId,
    );

    fn suggest_dereferences(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        points_at_arg: bool,
    );

    fn get_closure_name(
        &self,
        def_id: DefId,
        err: &mut DiagnosticBuilder<'_>,
        msg: &str,
    ) -> Option<String>;

    fn suggest_fn_call(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
    );

    fn suggest_add_reference_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
        has_custom_message: bool,
    ) -> bool;

    fn suggest_remove_reference(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
    );

    fn suggest_change_mut(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
    );

    fn suggest_semicolon_removal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
    );

    fn return_type_span(&self, obligation: &PredicateObligation<'tcx>) -> Option<Span>;

    fn suggest_impl_trait(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        obligation: &PredicateObligation<'tcx>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
    ) -> bool;

    fn point_at_returns_when_relevant(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
    );

    fn report_closure_arg_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_ref: ty::PolyTraitRef<'tcx>,
        found: ty::PolyTraitRef<'tcx>,
    ) -> DiagnosticBuilder<'tcx>;

    fn suggest_fully_qualified_path(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        def_id: DefId,
        span: Span,
        trait_ref: DefId,
    );

    fn maybe_note_obligation_cause_for_async_await(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool;

    fn note_obligation_cause_for_async_await(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        interior_or_upvar_span: GeneratorInteriorOrUpvar,
        interior_extra_info: Option<(Option<Span>, Span, Option<hir::HirId>, Option<Span>)>,
        inner_generator_body: Option<&hir::Body<'tcx>>,
        outer_generator: Option<DefId>,
        trait_ref: ty::TraitRef<'tcx>,
        target_ty: Ty<'tcx>,
        typeck_results: &ty::TypeckResults<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        next_code: Option<&ObligationCauseCode<'tcx>>,
    );

    fn note_obligation_cause_code<T>(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        predicate: &T,
        cause_code: &ObligationCauseCode<'tcx>,
        obligated_types: &mut Vec<&ty::TyS<'tcx>>,
        seen_requirements: &mut FxHashSet<DefId>,
    ) where
        T: fmt::Display;

    fn suggest_new_overflow_limit(&self, err: &mut DiagnosticBuilder<'_>);

    /// Suggest to await before try: future? => future.await?
    fn suggest_await_before_try(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
        span: Span,
    );
}

fn predicate_constraint(generics: &hir::Generics<'_>, pred: String) -> (Span, String) {
    (
        generics.where_clause.tail_span_for_suggestion(),
        format!(
            "{} {}",
            if !generics.where_clause.predicates.is_empty() { "," } else { " where" },
            pred,
        ),
    )
}

/// Type parameter needs more bounds. The trivial case is `T` `where T: Bound`, but
/// it can also be an `impl Trait` param that needs to be decomposed to a type
/// param for cleaner code.
fn suggest_restriction(
    tcx: TyCtxt<'tcx>,
    generics: &hir::Generics<'tcx>,
    msg: &str,
    err: &mut DiagnosticBuilder<'_>,
    fn_sig: Option<&hir::FnSig<'_>>,
    projection: Option<&ty::ProjectionTy<'_>>,
    trait_ref: ty::PolyTraitRef<'tcx>,
    super_traits: Option<(&Ident, &hir::GenericBounds<'_>)>,
) {
    // When we are dealing with a trait, `super_traits` will be `Some`:
    // Given `trait T: A + B + C {}`
    //              -  ^^^^^^^^^ GenericBounds
    //              |
    //              &Ident
    let span = generics.where_clause.span_for_predicates_or_empty_place();
    if span.from_expansion() || span.desugaring_kind().is_some() {
        return;
    }
    // Given `fn foo(t: impl Trait)` where `Trait` requires assoc type `A`...
    if let Some((bound_str, fn_sig)) =
        fn_sig.zip(projection).and_then(|(sig, p)| match p.self_ty().kind() {
            // Shenanigans to get the `Trait` from the `impl Trait`.
            ty::Param(param) => {
                // `fn foo(t: impl Trait)`
                //                 ^^^^^ get this string
                param.name.as_str().strip_prefix("impl").map(|s| (s.trim_start().to_string(), sig))
            }
            _ => None,
        })
    {
        // We know we have an `impl Trait` that doesn't satisfy a required projection.

        // Find all of the ocurrences of `impl Trait` for `Trait` in the function arguments'
        // types. There should be at least one, but there might be *more* than one. In that
        // case we could just ignore it and try to identify which one needs the restriction,
        // but instead we choose to suggest replacing all instances of `impl Trait` with `T`
        // where `T: Trait`.
        let mut ty_spans = vec![];
        let impl_trait_str = format!("impl {}", bound_str);
        for input in fn_sig.decl.inputs {
            if let hir::TyKind::Path(hir::QPath::Resolved(
                None,
                hir::Path { segments: [segment], .. },
            )) = input.kind
            {
                if segment.ident.as_str() == impl_trait_str.as_str() {
                    // `fn foo(t: impl Trait)`
                    //            ^^^^^^^^^^ get this to suggest `T` instead

                    // There might be more than one `impl Trait`.
                    ty_spans.push(input.span);
                }
            }
        }

        let type_param_name = generics.params.next_type_param_name(Some(&bound_str));
        // The type param `T: Trait` we will suggest to introduce.
        let type_param = format!("{}: {}", type_param_name, bound_str);

        // FIXME: modify the `trait_ref` instead of string shenanigans.
        // Turn `<impl Trait as Foo>::Bar: Qux` into `<T as Foo>::Bar: Qux`.
        let pred = trait_ref.without_const().to_predicate(tcx).to_string();
        let pred = pred.replace(&impl_trait_str, &type_param_name);
        let mut sugg = vec![
            match generics
                .params
                .iter()
                .filter(|p| match p.kind {
                    hir::GenericParamKind::Type {
                        synthetic: Some(hir::SyntheticTyParamKind::ImplTrait),
                        ..
                    } => false,
                    _ => true,
                })
                .last()
            {
                // `fn foo(t: impl Trait)`
                //        ^ suggest `<T: Trait>` here
                None => (generics.span, format!("<{}>", type_param)),
                // `fn foo<A>(t: impl Trait)`
                //        ^^^ suggest `<A, T: Trait>` here
                Some(param) => (
                    param.bounds_span().unwrap_or(param.span).shrink_to_hi(),
                    format!(", {}", type_param),
                ),
            },
            // `fn foo(t: impl Trait)`
            //                       ^ suggest `where <T as Trait>::A: Bound`
            predicate_constraint(generics, pred),
        ];
        sugg.extend(ty_spans.into_iter().map(|s| (s, type_param_name.to_string())));

        // Suggest `fn foo<T: Trait>(t: T) where <T as Trait>::A: Bound`.
        // FIXME: once `#![feature(associated_type_bounds)]` is stabilized, we should suggest
        // `fn foo(t: impl Trait<A: Bound>)` instead.
        err.multipart_suggestion(
            "introduce a type parameter with a trait bound instead of using `impl Trait`",
            sugg,
            Applicability::MaybeIncorrect,
        );
    } else {
        // Trivial case: `T` needs an extra bound: `T: Bound`.
        let (sp, suggestion) = match super_traits {
            None => predicate_constraint(
                generics,
                trait_ref.without_const().to_predicate(tcx).to_string(),
            ),
            Some((ident, bounds)) => match bounds {
                [.., bound] => (
                    bound.span().shrink_to_hi(),
                    format!(" + {}", trait_ref.print_only_trait_path().to_string()),
                ),
                [] => (
                    ident.span.shrink_to_hi(),
                    format!(": {}", trait_ref.print_only_trait_path().to_string()),
                ),
            },
        };

        err.span_suggestion_verbose(
            sp,
            &format!("consider further restricting {}", msg),
            suggestion,
            Applicability::MachineApplicable,
        );
    }
}

impl<'a, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'a, 'tcx> {
    fn suggest_restricting_param_bound(
        &self,
        mut err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        body_id: hir::HirId,
    ) {
        let self_ty = trait_ref.skip_binder().self_ty();
        let (param_ty, projection) = match self_ty.kind() {
            ty::Param(_) => (true, None),
            ty::Projection(projection) => (false, Some(projection)),
            _ => return,
        };

        // FIXME: Add check for trait bound that is already present, particularly `?Sized` so we
        //        don't suggest `T: Sized + ?Sized`.
        let mut hir_id = body_id;
        while let Some(node) = self.tcx.hir().find(hir_id) {
            match node {
                hir::Node::Item(hir::Item {
                    ident,
                    kind: hir::ItemKind::Trait(_, _, generics, bounds, _),
                    ..
                }) if self_ty == self.tcx.types.self_param => {
                    assert!(param_ty);
                    // Restricting `Self` for a single method.
                    suggest_restriction(
                        self.tcx,
                        &generics,
                        "`Self`",
                        err,
                        None,
                        projection,
                        trait_ref,
                        Some((ident, bounds)),
                    );
                    return;
                }

                hir::Node::TraitItem(hir::TraitItem {
                    generics,
                    kind: hir::TraitItemKind::Fn(..),
                    ..
                }) if self_ty == self.tcx.types.self_param => {
                    assert!(param_ty);
                    // Restricting `Self` for a single method.
                    suggest_restriction(
                        self.tcx, &generics, "`Self`", err, None, projection, trait_ref, None,
                    );
                    return;
                }

                hir::Node::TraitItem(hir::TraitItem {
                    generics,
                    kind: hir::TraitItemKind::Fn(fn_sig, ..),
                    ..
                })
                | hir::Node::ImplItem(hir::ImplItem {
                    generics,
                    kind: hir::ImplItemKind::Fn(fn_sig, ..),
                    ..
                })
                | hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Fn(fn_sig, generics, _), ..
                }) if projection.is_some() => {
                    // Missing restriction on associated type of type parameter (unmet projection).
                    suggest_restriction(
                        self.tcx,
                        &generics,
                        "the associated type",
                        err,
                        Some(fn_sig),
                        projection,
                        trait_ref,
                        None,
                    );
                    return;
                }
                hir::Node::Item(hir::Item {
                    kind:
                        hir::ItemKind::Trait(_, _, generics, _, _)
                        | hir::ItemKind::Impl { generics, .. },
                    ..
                }) if projection.is_some() => {
                    // Missing restriction on associated type of type parameter (unmet projection).
                    suggest_restriction(
                        self.tcx,
                        &generics,
                        "the associated type",
                        err,
                        None,
                        projection,
                        trait_ref,
                        None,
                    );
                    return;
                }

                hir::Node::Item(hir::Item {
                    kind:
                        hir::ItemKind::Struct(_, generics)
                        | hir::ItemKind::Enum(_, generics)
                        | hir::ItemKind::Union(_, generics)
                        | hir::ItemKind::Trait(_, _, generics, ..)
                        | hir::ItemKind::Impl { generics, .. }
                        | hir::ItemKind::Fn(_, generics, _)
                        | hir::ItemKind::TyAlias(_, generics)
                        | hir::ItemKind::TraitAlias(generics, _)
                        | hir::ItemKind::OpaqueTy(hir::OpaqueTy { generics, .. }),
                    ..
                })
                | hir::Node::TraitItem(hir::TraitItem { generics, .. })
                | hir::Node::ImplItem(hir::ImplItem { generics, .. })
                    if param_ty =>
                {
                    // Missing generic type parameter bound.
                    let param_name = self_ty.to_string();
                    let constraint = trait_ref.print_only_trait_path().to_string();
                    if suggest_constraining_type_param(
                        self.tcx,
                        generics,
                        &mut err,
                        &param_name,
                        &constraint,
                        Some(trait_ref.def_id()),
                    ) {
                        return;
                    }
                }

                hir::Node::Crate(..) => return,

                _ => {}
            }

            hir_id = self.tcx.hir().get_parent_item(hir_id);
        }
    }

    /// When after several dereferencing, the reference satisfies the trait
    /// binding. This function provides dereference suggestion for this
    /// specific situation.
    fn suggest_dereferences(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        points_at_arg: bool,
    ) {
        // It only make sense when suggesting dereferences for arguments
        if !points_at_arg {
            return;
        }
        let param_env = obligation.param_env;
        let body_id = obligation.cause.body_id;
        let span = obligation.cause.span;
        let real_trait_ref = match &obligation.cause.code {
            ObligationCauseCode::ImplDerivedObligation(cause)
            | ObligationCauseCode::DerivedObligation(cause)
            | ObligationCauseCode::BuiltinDerivedObligation(cause) => cause.parent_trait_ref,
            _ => trait_ref,
        };
        let real_ty = match real_trait_ref.self_ty().no_bound_vars() {
            Some(ty) => ty,
            None => return,
        };

        if let ty::Ref(region, base_ty, mutbl) = *real_ty.kind() {
            let mut autoderef = Autoderef::new(self, param_env, body_id, span, base_ty, span);
            if let Some(steps) = autoderef.find_map(|(ty, steps)| {
                // Re-add the `&`
                let ty = self.tcx.mk_ref(region, TypeAndMut { ty, mutbl });
                let obligation =
                    self.mk_trait_obligation_with_new_self_ty(param_env, real_trait_ref, ty);
                Some(steps).filter(|_| self.predicate_may_hold(&obligation))
            }) {
                if steps > 0 {
                    if let Ok(src) = self.tcx.sess.source_map().span_to_snippet(span) {
                        // Don't care about `&mut` because `DerefMut` is used less
                        // often and user will not expect autoderef happens.
                        if src.starts_with('&') && !src.starts_with("&mut ") {
                            let derefs = "*".repeat(steps);
                            err.span_suggestion(
                                span,
                                "consider adding dereference here",
                                format!("&{}{}", derefs, &src[1..]),
                                Applicability::MachineApplicable,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Given a closure's `DefId`, return the given name of the closure.
    ///
    /// This doesn't account for reassignments, but it's only used for suggestions.
    fn get_closure_name(
        &self,
        def_id: DefId,
        err: &mut DiagnosticBuilder<'_>,
        msg: &str,
    ) -> Option<String> {
        let get_name =
            |err: &mut DiagnosticBuilder<'_>, kind: &hir::PatKind<'_>| -> Option<String> {
                // Get the local name of this closure. This can be inaccurate because
                // of the possibility of reassignment, but this should be good enough.
                match &kind {
                    hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, name, None) => {
                        Some(format!("{}", name))
                    }
                    _ => {
                        err.note(&msg);
                        None
                    }
                }
            };

        let hir = self.tcx.hir();
        let hir_id = hir.local_def_id_to_hir_id(def_id.as_local()?);
        let parent_node = hir.get_parent_node(hir_id);
        match hir.find(parent_node) {
            Some(hir::Node::Stmt(hir::Stmt { kind: hir::StmtKind::Local(local), .. })) => {
                get_name(err, &local.pat.kind)
            }
            // Different to previous arm because one is `&hir::Local` and the other
            // is `P<hir::Local>`.
            Some(hir::Node::Local(local)) => get_name(err, &local.pat.kind),
            _ => None,
        }
    }

    /// We tried to apply the bound to an `fn` or closure. Check whether calling it would
    /// evaluate to a type that *would* satisfy the trait binding. If it would, suggest calling
    /// it: `bar(foo)` → `bar(foo())`. This case is *very* likely to be hit if `foo` is `async`.
    fn suggest_fn_call(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
    ) {
        let self_ty = match trait_ref.self_ty().no_bound_vars() {
            None => return,
            Some(ty) => ty,
        };

        let (def_id, output_ty, callable) = match *self_ty.kind() {
            ty::Closure(def_id, substs) => (def_id, substs.as_closure().sig().output(), "closure"),
            ty::FnDef(def_id, _) => (def_id, self_ty.fn_sig(self.tcx).output(), "function"),
            _ => return,
        };
        let msg = format!("use parentheses to call the {}", callable);

        // `mk_trait_obligation_with_new_self_ty` only works for types with no escaping bound
        // variables, so bail out if we have any.
        let output_ty = match output_ty.no_bound_vars() {
            Some(ty) => ty,
            None => return,
        };

        let new_obligation =
            self.mk_trait_obligation_with_new_self_ty(obligation.param_env, trait_ref, output_ty);

        match self.evaluate_obligation(&new_obligation) {
            Ok(
                EvaluationResult::EvaluatedToOk
                | EvaluationResult::EvaluatedToOkModuloRegions
                | EvaluationResult::EvaluatedToAmbig,
            ) => {}
            _ => return,
        }
        let hir = self.tcx.hir();
        // Get the name of the callable and the arguments to be used in the suggestion.
        let (snippet, sugg) = match hir.get_if_local(def_id) {
            Some(hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(_, decl, _, span, ..),
                ..
            })) => {
                err.span_label(*span, "consider calling this closure");
                let name = match self.get_closure_name(def_id, err, &msg) {
                    Some(name) => name,
                    None => return,
                };
                let args = decl.inputs.iter().map(|_| "_").collect::<Vec<_>>().join(", ");
                let sugg = format!("({})", args);
                (format!("{}{}", name, sugg), sugg)
            }
            Some(hir::Node::Item(hir::Item {
                ident,
                kind: hir::ItemKind::Fn(.., body_id),
                ..
            })) => {
                err.span_label(ident.span, "consider calling this function");
                let body = hir.body(*body_id);
                let args = body
                    .params
                    .iter()
                    .map(|arg| match &arg.pat.kind {
                        hir::PatKind::Binding(_, _, ident, None)
                        // FIXME: provide a better suggestion when encountering `SelfLower`, it
                        // should suggest a method call.
                        if ident.name != kw::SelfLower => ident.to_string(),
                        _ => "_".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let sugg = format!("({})", args);
                (format!("{}{}", ident, sugg), sugg)
            }
            _ => return,
        };
        if points_at_arg {
            // When the obligation error has been ensured to have been caused by
            // an argument, the `obligation.cause.span` points at the expression
            // of the argument, so we can provide a suggestion. This is signaled
            // by `points_at_arg`. Otherwise, we give a more general note.
            err.span_suggestion_verbose(
                obligation.cause.span.shrink_to_hi(),
                &msg,
                sugg,
                Applicability::HasPlaceholders,
            );
        } else {
            err.help(&format!("{}: `{}`", msg, snippet));
        }
    }

    fn suggest_add_reference_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
        has_custom_message: bool,
    ) -> bool {
        if !points_at_arg {
            return false;
        }

        let span = obligation.cause.span;
        let param_env = obligation.param_env;
        let trait_ref = trait_ref.skip_binder();

        if let ObligationCauseCode::ImplDerivedObligation(obligation) = &obligation.cause.code {
            // Try to apply the original trait binding obligation by borrowing.
            let self_ty = trait_ref.self_ty();
            let found = self_ty.to_string();
            let new_self_ty = self.tcx.mk_imm_ref(self.tcx.lifetimes.re_static, self_ty);
            let substs = self.tcx.mk_substs_trait(new_self_ty, &[]);
            let new_trait_ref = ty::TraitRef::new(obligation.parent_trait_ref.def_id(), substs);
            let new_obligation = Obligation::new(
                ObligationCause::dummy(),
                param_env,
                new_trait_ref.without_const().to_predicate(self.tcx),
            );

            if self.predicate_must_hold_modulo_regions(&new_obligation) {
                if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                    // We have a very specific type of error, where just borrowing this argument
                    // might solve the problem. In cases like this, the important part is the
                    // original type obligation, not the last one that failed, which is arbitrary.
                    // Because of this, we modify the error to refer to the original obligation and
                    // return early in the caller.

                    let msg = format!(
                        "the trait bound `{}: {}` is not satisfied",
                        found,
                        obligation.parent_trait_ref.skip_binder().print_only_trait_path(),
                    );
                    if has_custom_message {
                        err.note(&msg);
                    } else {
                        err.message = vec![(msg, Style::NoStyle)];
                    }
                    if snippet.starts_with('&') {
                        // This is already a literal borrow and the obligation is failing
                        // somewhere else in the obligation chain. Do not suggest non-sense.
                        return false;
                    }
                    err.span_label(
                        span,
                        &format!(
                            "expected an implementor of trait `{}`",
                            obligation.parent_trait_ref.skip_binder().print_only_trait_path(),
                        ),
                    );

                    // This if is to prevent a special edge-case
                    if !span.from_expansion() {
                        // We don't want a borrowing suggestion on the fields in structs,
                        // ```
                        // struct Foo {
                        //  the_foos: Vec<Foo>
                        // }
                        // ```

                        err.span_suggestion(
                            span,
                            "consider borrowing here",
                            format!("&{}", snippet),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    return true;
                }
            }
        }
        false
    }

    /// Whenever references are used by mistake, like `for (i, e) in &vec.iter().enumerate()`,
    /// suggest removing these references until we reach a type that implements the trait.
    fn suggest_remove_reference(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
    ) {
        let span = obligation.cause.span;

        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
            let refs_number =
                snippet.chars().filter(|c| !c.is_whitespace()).take_while(|c| *c == '&').count();
            if let Some('\'') = snippet.chars().filter(|c| !c.is_whitespace()).nth(refs_number) {
                // Do not suggest removal of borrow from type arguments.
                return;
            }

            let mut suggested_ty = match trait_ref.self_ty().no_bound_vars() {
                Some(ty) => ty,
                None => return,
            };

            for refs_remaining in 0..refs_number {
                if let ty::Ref(_, inner_ty, _) = suggested_ty.kind() {
                    suggested_ty = inner_ty;

                    let new_obligation = self.mk_trait_obligation_with_new_self_ty(
                        obligation.param_env,
                        trait_ref,
                        suggested_ty,
                    );

                    if self.predicate_may_hold(&new_obligation) {
                        let sp = self
                            .tcx
                            .sess
                            .source_map()
                            .span_take_while(span, |c| c.is_whitespace() || *c == '&');

                        let remove_refs = refs_remaining + 1;

                        let msg = if remove_refs == 1 {
                            "consider removing the leading `&`-reference".to_string()
                        } else {
                            format!("consider removing {} leading `&`-references", remove_refs)
                        };

                        err.span_suggestion_short(
                            sp,
                            &msg,
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    /// Check if the trait bound is implemented for a different mutability and note it in the
    /// final error.
    fn suggest_change_mut(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
    ) {
        let span = obligation.cause.span;
        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
            let refs_number =
                snippet.chars().filter(|c| !c.is_whitespace()).take_while(|c| *c == '&').count();
            if let Some('\'') = snippet.chars().filter(|c| !c.is_whitespace()).nth(refs_number) {
                // Do not suggest removal of borrow from type arguments.
                return;
            }
            let trait_ref = self.resolve_vars_if_possible(trait_ref);
            if trait_ref.has_infer_types_or_consts() {
                // Do not ICE while trying to find if a reborrow would succeed on a trait with
                // unresolved bindings.
                return;
            }

            if let ty::Ref(region, t_type, mutability) = *trait_ref.skip_binder().self_ty().kind() {
                if region.is_late_bound() || t_type.has_escaping_bound_vars() {
                    // Avoid debug assertion in `mk_obligation_for_def_id`.
                    //
                    // If the self type has escaping bound vars then it's not
                    // going to be the type of an expression, so the suggestion
                    // probably won't apply anyway.
                    return;
                }

                let suggested_ty = match mutability {
                    hir::Mutability::Mut => self.tcx.mk_imm_ref(region, t_type),
                    hir::Mutability::Not => self.tcx.mk_mut_ref(region, t_type),
                };

                let new_obligation = self.mk_trait_obligation_with_new_self_ty(
                    obligation.param_env,
                    trait_ref,
                    suggested_ty,
                );
                let suggested_ty_would_satisfy_obligation = self
                    .evaluate_obligation_no_overflow(&new_obligation)
                    .must_apply_modulo_regions();
                if suggested_ty_would_satisfy_obligation {
                    let sp = self
                        .tcx
                        .sess
                        .source_map()
                        .span_take_while(span, |c| c.is_whitespace() || *c == '&');
                    if points_at_arg && mutability == hir::Mutability::Not && refs_number > 0 {
                        err.span_suggestion_verbose(
                            sp,
                            "consider changing this borrow's mutability",
                            "&mut ".to_string(),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.note(&format!(
                            "`{}` is implemented for `{:?}`, but not for `{:?}`",
                            trait_ref.print_only_trait_path(),
                            suggested_ty,
                            trait_ref.skip_binder().self_ty(),
                        ));
                    }
                }
            }
        }
    }

    fn suggest_semicolon_removal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
    ) {
        let is_empty_tuple =
            |ty: ty::Binder<Ty<'_>>| *ty.skip_binder().kind() == ty::Tuple(ty::List::empty());

        let hir = self.tcx.hir();
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let node = hir.find(parent_node);
        if let Some(hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(sig, _, body_id), ..
        })) = node
        {
            let body = hir.body(*body_id);
            if let hir::ExprKind::Block(blk, _) = &body.value.kind {
                if sig.decl.output.span().overlaps(span)
                    && blk.expr.is_none()
                    && is_empty_tuple(trait_ref.self_ty())
                {
                    // FIXME(estebank): When encountering a method with a trait
                    // bound not satisfied in the return type with a body that has
                    // no return, suggest removal of semicolon on last statement.
                    // Once that is added, close #54771.
                    if let Some(ref stmt) = blk.stmts.last() {
                        let sp = self.tcx.sess.source_map().end_point(stmt.span);
                        err.span_label(sp, "consider removing this semicolon");
                    }
                }
            }
        }
    }

    fn return_type_span(&self, obligation: &PredicateObligation<'tcx>) -> Option<Span> {
        let hir = self.tcx.hir();
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let sig = match hir.find(parent_node) {
            Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, ..), .. })) => sig,
            _ => return None,
        };

        if let hir::FnRetTy::Return(ret_ty) = sig.decl.output { Some(ret_ty.span) } else { None }
    }

    /// If all conditions are met to identify a returned `dyn Trait`, suggest using `impl Trait` if
    /// applicable and signal that the error has been expanded appropriately and needs to be
    /// emitted.
    fn suggest_impl_trait(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        obligation: &PredicateObligation<'tcx>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
    ) -> bool {
        match obligation.cause.code.peel_derives() {
            // Only suggest `impl Trait` if the return type is unsized because it is `dyn Trait`.
            ObligationCauseCode::SizedReturnType => {}
            _ => return false,
        }

        let hir = self.tcx.hir();
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let node = hir.find(parent_node);
        let (sig, body_id) = if let Some(hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(sig, _, body_id),
            ..
        })) = node
        {
            (sig, body_id)
        } else {
            return false;
        };
        let body = hir.body(*body_id);
        let trait_ref = self.resolve_vars_if_possible(trait_ref);
        let ty = trait_ref.skip_binder().self_ty();
        let is_object_safe = match ty.kind() {
            ty::Dynamic(predicates, _) => {
                // If the `dyn Trait` is not object safe, do not suggest `Box<dyn Trait>`.
                predicates
                    .principal_def_id()
                    .map_or(true, |def_id| self.tcx.object_safety_violations(def_id).is_empty())
            }
            // We only want to suggest `impl Trait` to `dyn Trait`s.
            // For example, `fn foo() -> str` needs to be filtered out.
            _ => return false,
        };

        let ret_ty = if let hir::FnRetTy::Return(ret_ty) = sig.decl.output {
            ret_ty
        } else {
            return false;
        };

        // Use `TypeVisitor` instead of the output type directly to find the span of `ty` for
        // cases like `fn foo() -> (dyn Trait, i32) {}`.
        // Recursively look for `TraitObject` types and if there's only one, use that span to
        // suggest `impl Trait`.

        // Visit to make sure there's a single `return` type to suggest `impl Trait`,
        // otherwise suggest using `Box<dyn Trait>` or an enum.
        let mut visitor = ReturnsVisitor::default();
        visitor.visit_body(&body);

        let typeck_results = self.in_progress_typeck_results.map(|t| t.borrow()).unwrap();

        let mut ret_types = visitor
            .returns
            .iter()
            .filter_map(|expr| typeck_results.node_type_opt(expr.hir_id))
            .map(|ty| self.resolve_vars_if_possible(ty));
        let (last_ty, all_returns_have_same_type, only_never_return) = ret_types.clone().fold(
            (None, true, true),
            |(last_ty, mut same, only_never_return): (std::option::Option<Ty<'_>>, bool, bool),
             ty| {
                let ty = self.resolve_vars_if_possible(ty);
                same &=
                    !matches!(ty.kind(), ty::Error(_))
                        && last_ty.map_or(true, |last_ty| {
                            // FIXME: ideally we would use `can_coerce` here instead, but `typeck` comes
                            // *after* in the dependency graph.
                            match (ty.kind(), last_ty.kind()) {
                                (Infer(InferTy::IntVar(_)), Infer(InferTy::IntVar(_)))
                                | (Infer(InferTy::FloatVar(_)), Infer(InferTy::FloatVar(_)))
                                | (Infer(InferTy::FreshIntTy(_)), Infer(InferTy::FreshIntTy(_)))
                                | (
                                    Infer(InferTy::FreshFloatTy(_)),
                                    Infer(InferTy::FreshFloatTy(_)),
                                ) => true,
                                _ => ty == last_ty,
                            }
                        });
                (Some(ty), same, only_never_return && matches!(ty.kind(), ty::Never))
            },
        );
        let all_returns_conform_to_trait =
            if let Some(ty_ret_ty) = typeck_results.node_type_opt(ret_ty.hir_id) {
                match ty_ret_ty.kind() {
                    ty::Dynamic(predicates, _) => {
                        let cause = ObligationCause::misc(ret_ty.span, ret_ty.hir_id);
                        let param_env = ty::ParamEnv::empty();
                        only_never_return
                            || ret_types.all(|returned_ty| {
                                predicates.iter().all(|predicate| {
                                    let pred = predicate.with_self_ty(self.tcx, returned_ty);
                                    let obl = Obligation::new(cause.clone(), param_env, pred);
                                    self.predicate_may_hold(&obl)
                                })
                            })
                    }
                    _ => false,
                }
            } else {
                true
            };

        let sm = self.tcx.sess.source_map();
        let snippet = if let (true, hir::TyKind::TraitObject(..), Ok(snippet), true) = (
            // Verify that we're dealing with a return `dyn Trait`
            ret_ty.span.overlaps(span),
            &ret_ty.kind,
            sm.span_to_snippet(ret_ty.span),
            // If any of the return types does not conform to the trait, then we can't
            // suggest `impl Trait` nor trait objects: it is a type mismatch error.
            all_returns_conform_to_trait,
        ) {
            snippet
        } else {
            return false;
        };
        err.code(error_code!(E0746));
        err.set_primary_message("return type cannot have an unboxed trait object");
        err.children.clear();
        let impl_trait_msg = "for information on `impl Trait`, see \
            <https://doc.rust-lang.org/book/ch10-02-traits.html\
            #returning-types-that-implement-traits>";
        let trait_obj_msg = "for information on trait objects, see \
            <https://doc.rust-lang.org/book/ch17-02-trait-objects.html\
            #using-trait-objects-that-allow-for-values-of-different-types>";
        let has_dyn = snippet.split_whitespace().next().map_or(false, |s| s == "dyn");
        let trait_obj = if has_dyn { &snippet[4..] } else { &snippet[..] };
        if only_never_return {
            // No return paths, probably using `panic!()` or similar.
            // Suggest `-> T`, `-> impl Trait`, and if `Trait` is object safe, `-> Box<dyn Trait>`.
            suggest_trait_object_return_type_alternatives(
                err,
                ret_ty.span,
                trait_obj,
                is_object_safe,
            );
        } else if let (Some(last_ty), true) = (last_ty, all_returns_have_same_type) {
            // Suggest `-> impl Trait`.
            err.span_suggestion(
                ret_ty.span,
                &format!(
                    "use `impl {1}` as the return type, as all return paths are of type `{}`, \
                     which implements `{1}`",
                    last_ty, trait_obj,
                ),
                format!("impl {}", trait_obj),
                Applicability::MachineApplicable,
            );
            err.note(impl_trait_msg);
        } else {
            if is_object_safe {
                // Suggest `-> Box<dyn Trait>` and `Box::new(returned_value)`.
                // Get all the return values and collect their span and suggestion.
                if let Some(mut suggestions) = visitor
                    .returns
                    .iter()
                    .map(|expr| {
                        let snip = sm.span_to_snippet(expr.span).ok()?;
                        Some((expr.span, format!("Box::new({})", snip)))
                    })
                    .collect::<Option<Vec<_>>>()
                {
                    // Add the suggestion for the return type.
                    suggestions.push((ret_ty.span, format!("Box<dyn {}>", trait_obj)));
                    err.multipart_suggestion(
                        "return a boxed trait object instead",
                        suggestions,
                        Applicability::MaybeIncorrect,
                    );
                }
            } else {
                // This is currently not possible to trigger because E0038 takes precedence, but
                // leave it in for completeness in case anything changes in an earlier stage.
                err.note(&format!(
                    "if trait `{}` was object safe, you could return a trait object",
                    trait_obj,
                ));
            }
            err.note(trait_obj_msg);
            err.note(&format!(
                "if all the returned values were of the same type you could use `impl {}` as the \
                 return type",
                trait_obj,
            ));
            err.note(impl_trait_msg);
            err.note("you can create a new `enum` with a variant for each returned type");
        }
        true
    }

    fn point_at_returns_when_relevant(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        match obligation.cause.code.peel_derives() {
            ObligationCauseCode::SizedReturnType => {}
            _ => return,
        }

        let hir = self.tcx.hir();
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let node = hir.find(parent_node);
        if let Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(_, _, body_id), .. })) =
            node
        {
            let body = hir.body(*body_id);
            // Point at all the `return`s in the function as they have failed trait bounds.
            let mut visitor = ReturnsVisitor::default();
            visitor.visit_body(&body);
            let typeck_results = self.in_progress_typeck_results.map(|t| t.borrow()).unwrap();
            for expr in &visitor.returns {
                if let Some(returned_ty) = typeck_results.node_type_opt(expr.hir_id) {
                    let ty = self.resolve_vars_if_possible(returned_ty);
                    err.span_label(expr.span, &format!("this returned value is of type `{}`", ty));
                }
            }
        }
    }

    fn report_closure_arg_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_ref: ty::PolyTraitRef<'tcx>,
        found: ty::PolyTraitRef<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        crate fn build_fn_sig_string<'tcx>(
            tcx: TyCtxt<'tcx>,
            trait_ref: ty::TraitRef<'tcx>,
        ) -> String {
            let inputs = trait_ref.substs.type_at(1);
            let sig = if let ty::Tuple(inputs) = inputs.kind() {
                tcx.mk_fn_sig(
                    inputs.iter().map(|k| k.expect_ty()),
                    tcx.mk_ty_infer(ty::TyVar(ty::TyVid { index: 0 })),
                    false,
                    hir::Unsafety::Normal,
                    abi::Abi::Rust,
                )
            } else {
                tcx.mk_fn_sig(
                    std::iter::once(inputs),
                    tcx.mk_ty_infer(ty::TyVar(ty::TyVid { index: 0 })),
                    false,
                    hir::Unsafety::Normal,
                    abi::Abi::Rust,
                )
            };
            ty::Binder::bind(sig).to_string()
        }

        let argument_is_closure = expected_ref.skip_binder().substs.type_at(0).is_closure();
        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0631,
            "type mismatch in {} arguments",
            if argument_is_closure { "closure" } else { "function" }
        );

        let found_str = format!(
            "expected signature of `{}`",
            build_fn_sig_string(self.tcx, found.skip_binder())
        );
        err.span_label(span, found_str);

        let found_span = found_span.unwrap_or(span);
        let expected_str = format!(
            "found signature of `{}`",
            build_fn_sig_string(self.tcx, expected_ref.skip_binder())
        );
        err.span_label(found_span, expected_str);

        err
    }

    fn suggest_fully_qualified_path(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        def_id: DefId,
        span: Span,
        trait_ref: DefId,
    ) {
        if let Some(assoc_item) = self.tcx.opt_associated_item(def_id) {
            if let ty::AssocKind::Const | ty::AssocKind::Type = assoc_item.kind {
                err.note(&format!(
                    "{}s cannot be accessed directly on a `trait`, they can only be \
                        accessed through a specific `impl`",
                    assoc_item.kind.as_def_kind().descr(def_id)
                ));
                err.span_suggestion(
                    span,
                    "use the fully qualified path to an implementation",
                    format!("<Type as {}>::{}", self.tcx.def_path_str(trait_ref), assoc_item.ident),
                    Applicability::HasPlaceholders,
                );
            }
        }
    }

    /// Adds an async-await specific note to the diagnostic when the future does not implement
    /// an auto trait because of a captured type.
    ///
    /// ```text
    /// note: future does not implement `Qux` as this value is used across an await
    ///   --> $DIR/issue-64130-3-other.rs:17:5
    ///    |
    /// LL |     let x = Foo;
    ///    |         - has type `Foo`
    /// LL |     baz().await;
    ///    |     ^^^^^^^^^^^ await occurs here, with `x` maybe used later
    /// LL | }
    ///    | - `x` is later dropped here
    /// ```
    ///
    /// When the diagnostic does not implement `Send` or `Sync` specifically, then the diagnostic
    /// is "replaced" with a different message and a more specific error.
    ///
    /// ```text
    /// error: future cannot be sent between threads safely
    ///   --> $DIR/issue-64130-2-send.rs:21:5
    ///    |
    /// LL | fn is_send<T: Send>(t: T) { }
    ///    |               ---- required by this bound in `is_send`
    /// ...
    /// LL |     is_send(bar());
    ///    |     ^^^^^^^ future returned by `bar` is not send
    ///    |
    ///    = help: within `impl std::future::Future`, the trait `std::marker::Send` is not
    ///            implemented for `Foo`
    /// note: future is not send as this value is used across an await
    ///   --> $DIR/issue-64130-2-send.rs:15:5
    ///    |
    /// LL |     let x = Foo;
    ///    |         - has type `Foo`
    /// LL |     baz().await;
    ///    |     ^^^^^^^^^^^ await occurs here, with `x` maybe used later
    /// LL | }
    ///    | - `x` is later dropped here
    /// ```
    ///
    /// Returns `true` if an async-await specific note was added to the diagnostic.
    fn maybe_note_obligation_cause_for_async_await(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool {
        debug!(
            "maybe_note_obligation_cause_for_async_await: obligation.predicate={:?} \
                obligation.cause.span={:?}",
            obligation.predicate, obligation.cause.span
        );
        let hir = self.tcx.hir();

        // Attempt to detect an async-await error by looking at the obligation causes, looking
        // for a generator to be present.
        //
        // When a future does not implement a trait because of a captured type in one of the
        // generators somewhere in the call stack, then the result is a chain of obligations.
        //
        // Given a `async fn` A that calls a `async fn` B which captures a non-send type and that
        // future is passed as an argument to a function C which requires a `Send` type, then the
        // chain looks something like this:
        //
        // - `BuiltinDerivedObligation` with a generator witness (B)
        // - `BuiltinDerivedObligation` with a generator (B)
        // - `BuiltinDerivedObligation` with `std::future::GenFuture` (B)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (B)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (B)
        // - `BuiltinDerivedObligation` with a generator witness (A)
        // - `BuiltinDerivedObligation` with a generator (A)
        // - `BuiltinDerivedObligation` with `std::future::GenFuture` (A)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (A)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (A)
        // - `BindingObligation` with `impl_send (Send requirement)
        //
        // The first obligation in the chain is the most useful and has the generator that captured
        // the type. The last generator (`outer_generator` below) has information about where the
        // bound was introduced. At least one generator should be present for this diagnostic to be
        // modified.
        let (mut trait_ref, mut target_ty) = match obligation.predicate.skip_binders() {
            ty::PredicateAtom::Trait(p, _) => (Some(p.trait_ref), Some(p.self_ty())),
            _ => (None, None),
        };
        let mut generator = None;
        let mut outer_generator = None;
        let mut next_code = Some(&obligation.cause.code);

        let mut seen_upvar_tys_infer_tuple = false;

        while let Some(code) = next_code {
            debug!("maybe_note_obligation_cause_for_async_await: code={:?}", code);
            match code {
                ObligationCauseCode::DerivedObligation(derived_obligation)
                | ObligationCauseCode::BuiltinDerivedObligation(derived_obligation)
                | ObligationCauseCode::ImplDerivedObligation(derived_obligation) => {
                    let ty = derived_obligation.parent_trait_ref.skip_binder().self_ty();
                    debug!(
                        "maybe_note_obligation_cause_for_async_await: \
                            parent_trait_ref={:?} self_ty.kind={:?}",
                        derived_obligation.parent_trait_ref,
                        ty.kind()
                    );

                    match *ty.kind() {
                        ty::Generator(did, ..) => {
                            generator = generator.or(Some(did));
                            outer_generator = Some(did);
                        }
                        ty::GeneratorWitness(..) => {}
                        ty::Tuple(_) if !seen_upvar_tys_infer_tuple => {
                            // By introducing a tuple of upvar types into the chain of obligations
                            // of a generator, the first non-generator item is now the tuple itself,
                            // we shall ignore this.

                            seen_upvar_tys_infer_tuple = true;
                        }
                        _ if generator.is_none() => {
                            trait_ref = Some(derived_obligation.parent_trait_ref.skip_binder());
                            target_ty = Some(ty);
                        }
                        _ => {}
                    }

                    next_code = Some(derived_obligation.parent_code.as_ref());
                }
                _ => break,
            }
        }

        // Only continue if a generator was found.
        debug!(
            "maybe_note_obligation_cause_for_async_await: generator={:?} trait_ref={:?} \
                target_ty={:?}",
            generator, trait_ref, target_ty
        );
        let (generator_did, trait_ref, target_ty) = match (generator, trait_ref, target_ty) {
            (Some(generator_did), Some(trait_ref), Some(target_ty)) => {
                (generator_did, trait_ref, target_ty)
            }
            _ => return false,
        };

        let span = self.tcx.def_span(generator_did);

        // Do not ICE on closure typeck (#66868).
        if !generator_did.is_local() {
            return false;
        }

        // Get the typeck results from the infcx if the generator is the function we are
        // currently type-checking; otherwise, get them by performing a query.
        // This is needed to avoid cycles.
        let in_progress_typeck_results = self.in_progress_typeck_results.map(|t| t.borrow());
        let generator_did_root = self.tcx.closure_base_def_id(generator_did);
        debug!(
            "maybe_note_obligation_cause_for_async_await: generator_did={:?} \
             generator_did_root={:?} in_progress_typeck_results.hir_owner={:?} span={:?}",
            generator_did,
            generator_did_root,
            in_progress_typeck_results.as_ref().map(|t| t.hir_owner),
            span
        );
        let query_typeck_results;
        let typeck_results: &TypeckResults<'tcx> = match &in_progress_typeck_results {
            Some(t) if t.hir_owner.to_def_id() == generator_did_root => t,
            _ => {
                query_typeck_results = self.tcx.typeck(generator_did.expect_local());
                &query_typeck_results
            }
        };

        let generator_body = generator_did
            .as_local()
            .map(|def_id| hir.local_def_id_to_hir_id(def_id))
            .and_then(|hir_id| hir.maybe_body_owned_by(hir_id))
            .map(|body_id| hir.body(body_id));
        let mut visitor = AwaitsVisitor::default();
        if let Some(body) = generator_body {
            visitor.visit_body(body);
        }
        debug!("maybe_note_obligation_cause_for_async_await: awaits = {:?}", visitor.awaits);

        // Look for a type inside the generator interior that matches the target type to get
        // a span.
        let target_ty_erased = self.tcx.erase_regions(target_ty);
        let ty_matches = |ty| -> bool {
            // Careful: the regions for types that appear in the
            // generator interior are not generally known, so we
            // want to erase them when comparing (and anyway,
            // `Send` and other bounds are generally unaffected by
            // the choice of region).  When erasing regions, we
            // also have to erase late-bound regions. This is
            // because the types that appear in the generator
            // interior generally contain "bound regions" to
            // represent regions that are part of the suspended
            // generator frame. Bound regions are preserved by
            // `erase_regions` and so we must also call
            // `erase_late_bound_regions`.
            let ty_erased = self.tcx.erase_late_bound_regions(ty::Binder::bind(ty));
            let ty_erased = self.tcx.erase_regions(ty_erased);
            let eq = ty::TyS::same_type(ty_erased, target_ty_erased);
            debug!(
                "maybe_note_obligation_cause_for_async_await: ty_erased={:?} \
                    target_ty_erased={:?} eq={:?}",
                ty_erased, target_ty_erased, eq
            );
            eq
        };

        let mut interior_or_upvar_span = None;
        let mut interior_extra_info = None;

        if let Some(upvars) = self.tcx.upvars_mentioned(generator_did) {
            interior_or_upvar_span = upvars.iter().find_map(|(upvar_id, upvar)| {
                let upvar_ty = typeck_results.node_type(*upvar_id);
                let upvar_ty = self.resolve_vars_if_possible(upvar_ty);
                if ty_matches(&upvar_ty) {
                    Some(GeneratorInteriorOrUpvar::Upvar(upvar.span))
                } else {
                    None
                }
            });
        };

        typeck_results
            .generator_interior_types
            .iter()
            .find(|ty::GeneratorInteriorTypeCause { ty, .. }| ty_matches(ty))
            .map(|cause| {
                // Check to see if any awaited expressions have the target type.
                let from_awaited_ty = visitor
                    .awaits
                    .into_iter()
                    .map(|id| hir.expect_expr(id))
                    .find(|await_expr| {
                        let ty = typeck_results.expr_ty_adjusted(&await_expr);
                        debug!(
                            "maybe_note_obligation_cause_for_async_await: await_expr={:?}",
                            await_expr
                        );
                        ty_matches(ty)
                    })
                    .map(|expr| expr.span);
                let ty::GeneratorInteriorTypeCause { span, scope_span, yield_span, expr, .. } =
                    cause;

                interior_or_upvar_span = Some(GeneratorInteriorOrUpvar::Interior(*span));
                interior_extra_info = Some((*scope_span, *yield_span, *expr, from_awaited_ty));
            });

        debug!(
            "maybe_note_obligation_cause_for_async_await: interior_or_upvar={:?} \
                generator_interior_types={:?}",
            interior_or_upvar_span, typeck_results.generator_interior_types
        );
        if let Some(interior_or_upvar_span) = interior_or_upvar_span {
            self.note_obligation_cause_for_async_await(
                err,
                interior_or_upvar_span,
                interior_extra_info,
                generator_body,
                outer_generator,
                trait_ref,
                target_ty,
                typeck_results,
                obligation,
                next_code,
            );
            true
        } else {
            false
        }
    }

    /// Unconditionally adds the diagnostic note described in
    /// `maybe_note_obligation_cause_for_async_await`'s documentation comment.
    fn note_obligation_cause_for_async_await(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        interior_or_upvar_span: GeneratorInteriorOrUpvar,
        interior_extra_info: Option<(Option<Span>, Span, Option<hir::HirId>, Option<Span>)>,
        inner_generator_body: Option<&hir::Body<'tcx>>,
        outer_generator: Option<DefId>,
        trait_ref: ty::TraitRef<'tcx>,
        target_ty: Ty<'tcx>,
        typeck_results: &ty::TypeckResults<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        next_code: Option<&ObligationCauseCode<'tcx>>,
    ) {
        let source_map = self.tcx.sess.source_map();

        let is_async = inner_generator_body
            .and_then(|body| body.generator_kind())
            .map(|generator_kind| matches!(generator_kind, hir::GeneratorKind::Async(..)))
            .unwrap_or(false);
        let (await_or_yield, an_await_or_yield) =
            if is_async { ("await", "an await") } else { ("yield", "a yield") };
        let future_or_generator = if is_async { "future" } else { "generator" };

        // Special case the primary error message when send or sync is the trait that was
        // not implemented.
        let is_send = self.tcx.is_diagnostic_item(sym::send_trait, trait_ref.def_id);
        let is_sync = self.tcx.is_diagnostic_item(sym::sync_trait, trait_ref.def_id);
        let hir = self.tcx.hir();
        let trait_explanation = if is_send || is_sync {
            let (trait_name, trait_verb) =
                if is_send { ("`Send`", "sent") } else { ("`Sync`", "shared") };

            err.clear_code();
            err.set_primary_message(format!(
                "{} cannot be {} between threads safely",
                future_or_generator, trait_verb
            ));

            let original_span = err.span.primary_span().unwrap();
            let mut span = MultiSpan::from_span(original_span);

            let message = outer_generator
                .and_then(|generator_did| {
                    Some(match self.tcx.generator_kind(generator_did).unwrap() {
                        GeneratorKind::Gen => format!("generator is not {}", trait_name),
                        GeneratorKind::Async(AsyncGeneratorKind::Fn) => self
                            .tcx
                            .parent(generator_did)
                            .and_then(|parent_did| parent_did.as_local())
                            .map(|parent_did| hir.local_def_id_to_hir_id(parent_did))
                            .and_then(|parent_hir_id| hir.opt_name(parent_hir_id))
                            .map(|name| {
                                format!("future returned by `{}` is not {}", name, trait_name)
                            })?,
                        GeneratorKind::Async(AsyncGeneratorKind::Block) => {
                            format!("future created by async block is not {}", trait_name)
                        }
                        GeneratorKind::Async(AsyncGeneratorKind::Closure) => {
                            format!("future created by async closure is not {}", trait_name)
                        }
                    })
                })
                .unwrap_or_else(|| format!("{} is not {}", future_or_generator, trait_name));

            span.push_span_label(original_span, message);
            err.set_span(span);

            format!("is not {}", trait_name)
        } else {
            format!("does not implement `{}`", trait_ref.print_only_trait_path())
        };

        let mut explain_yield =
            |interior_span: Span, yield_span: Span, scope_span: Option<Span>| {
                let mut span = MultiSpan::from_span(yield_span);
                if let Ok(snippet) = source_map.span_to_snippet(interior_span) {
                    // #70935: If snippet contains newlines, display "the value" instead
                    // so that we do not emit complex diagnostics.
                    let snippet = &format!("`{}`", snippet);
                    let snippet = if snippet.contains('\n') { "the value" } else { snippet };
                    // The multispan can be complex here, like:
                    // note: future is not `Send` as this value is used across an await
                    //   --> $DIR/issue-70935-complex-spans.rs:13:9
                    //    |
                    // LL |            baz(|| async{
                    //    |  __________^___-
                    //    | | _________|
                    //    | ||
                    // LL | ||             foo(tx.clone());
                    // LL | ||         }).await;
                    //    | ||         -      ^- value is later dropped here
                    //    | ||_________|______|
                    //    | |__________|      await occurs here, with value maybe used later
                    //    |            has type `closure` which is not `Send`
                    //
                    // So, detect it and separate into some notes, like:
                    //
                    // note: future is not `Send` as this value is used across an await
                    //   --> $DIR/issue-70935-complex-spans.rs:13:9
                    //    |
                    // LL | /         baz(|| async{
                    // LL | |             foo(tx.clone());
                    // LL | |         }).await;
                    //    | |________________^ first, await occurs here, with the value maybe used later...
                    // note: the value is later dropped here
                    //   --> $DIR/issue-70935-complex-spans.rs:15:17
                    //    |
                    // LL |         }).await;
                    //    |                 ^
                    //
                    // If available, use the scope span to annotate the drop location.
                    if let Some(scope_span) = scope_span {
                        let scope_span = source_map.end_point(scope_span);
                        let is_overlapped =
                            yield_span.overlaps(scope_span) || yield_span.overlaps(interior_span);
                        if is_overlapped {
                            span.push_span_label(
                                yield_span,
                                format!(
                                    "first, {} occurs here, with {} maybe used later...",
                                    await_or_yield, snippet
                                ),
                            );
                            err.span_note(
                                span,
                                &format!(
                                    "{} {} as this value is used across {}",
                                    future_or_generator, trait_explanation, an_await_or_yield
                                ),
                            );
                            if source_map.is_multiline(interior_span) {
                                err.span_note(
                                    scope_span,
                                    &format!("{} is later dropped here", snippet),
                                );
                                err.span_note(
                                    interior_span,
                                    &format!(
                                        "this has type `{}` which {}",
                                        target_ty, trait_explanation
                                    ),
                                );
                            } else {
                                let mut span = MultiSpan::from_span(scope_span);
                                span.push_span_label(
                                    interior_span,
                                    format!("has type `{}` which {}", target_ty, trait_explanation),
                                );
                                err.span_note(span, &format!("{} is later dropped here", snippet));
                            }
                        } else {
                            span.push_span_label(
                                yield_span,
                                format!(
                                    "{} occurs here, with {} maybe used later",
                                    await_or_yield, snippet
                                ),
                            );
                            span.push_span_label(
                                scope_span,
                                format!("{} is later dropped here", snippet),
                            );
                            span.push_span_label(
                                interior_span,
                                format!("has type `{}` which {}", target_ty, trait_explanation),
                            );
                            err.span_note(
                                span,
                                &format!(
                                    "{} {} as this value is used across {}",
                                    future_or_generator, trait_explanation, an_await_or_yield
                                ),
                            );
                        }
                    } else {
                        span.push_span_label(
                            yield_span,
                            format!(
                                "{} occurs here, with {} maybe used later",
                                await_or_yield, snippet
                            ),
                        );
                        span.push_span_label(
                            interior_span,
                            format!("has type `{}` which {}", target_ty, trait_explanation),
                        );
                        err.span_note(
                            span,
                            &format!(
                                "{} {} as this value is used across {}",
                                future_or_generator, trait_explanation, an_await_or_yield
                            ),
                        );
                    }
                }
            };
        match interior_or_upvar_span {
            GeneratorInteriorOrUpvar::Interior(interior_span) => {
                if let Some((scope_span, yield_span, expr, from_awaited_ty)) = interior_extra_info {
                    if let Some(await_span) = from_awaited_ty {
                        // The type causing this obligation is one being awaited at await_span.
                        let mut span = MultiSpan::from_span(await_span);
                        span.push_span_label(
                            await_span,
                            format!(
                                "await occurs here on type `{}`, which {}",
                                target_ty, trait_explanation
                            ),
                        );
                        err.span_note(
                            span,
                            &format!(
                                "future {not_trait} as it awaits another future which {not_trait}",
                                not_trait = trait_explanation
                            ),
                        );
                    } else {
                        // Look at the last interior type to get a span for the `.await`.
                        debug!(
                            "note_obligation_cause_for_async_await generator_interior_types: {:#?}",
                            typeck_results.generator_interior_types
                        );
                        explain_yield(interior_span, yield_span, scope_span);
                    }

                    if let Some(expr_id) = expr {
                        let expr = hir.expect_expr(expr_id);
                        debug!("target_ty evaluated from {:?}", expr);

                        let parent = hir.get_parent_node(expr_id);
                        if let Some(hir::Node::Expr(e)) = hir.find(parent) {
                            let parent_span = hir.span(parent);
                            let parent_did = parent.owner.to_def_id();
                            // ```rust
                            // impl T {
                            //     fn foo(&self) -> i32 {}
                            // }
                            // T.foo();
                            // ^^^^^^^ a temporary `&T` created inside this method call due to `&self`
                            // ```
                            //
                            let is_region_borrow = typeck_results
                                .expr_adjustments(expr)
                                .iter()
                                .any(|adj| adj.is_region_borrow());

                            // ```rust
                            // struct Foo(*const u8);
                            // bar(Foo(std::ptr::null())).await;
                            //     ^^^^^^^^^^^^^^^^^^^^^ raw-ptr `*T` created inside this struct ctor.
                            // ```
                            debug!("parent_def_kind: {:?}", self.tcx.def_kind(parent_did));
                            let is_raw_borrow_inside_fn_like_call =
                                match self.tcx.def_kind(parent_did) {
                                    DefKind::Fn | DefKind::Ctor(..) => target_ty.is_unsafe_ptr(),
                                    _ => false,
                                };

                            if (typeck_results.is_method_call(e) && is_region_borrow)
                                || is_raw_borrow_inside_fn_like_call
                            {
                                err.span_help(
                                    parent_span,
                                    "consider moving this into a `let` \
                        binding to create a shorter lived borrow",
                                );
                            }
                        }
                    }
                }
            }
            GeneratorInteriorOrUpvar::Upvar(upvar_span) => {
                let mut span = MultiSpan::from_span(upvar_span);
                span.push_span_label(
                    upvar_span,
                    format!("has type `{}` which {}", target_ty, trait_explanation),
                );
                err.span_note(span, &format!("captured value {}", trait_explanation));
            }
        }

        // Add a note for the item obligation that remains - normally a note pointing to the
        // bound that introduced the obligation (e.g. `T: Send`).
        debug!("note_obligation_cause_for_async_await: next_code={:?}", next_code);
        self.note_obligation_cause_code(
            err,
            &obligation.predicate,
            next_code.unwrap(),
            &mut Vec::new(),
            &mut Default::default(),
        );
    }

    fn note_obligation_cause_code<T>(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        predicate: &T,
        cause_code: &ObligationCauseCode<'tcx>,
        obligated_types: &mut Vec<&ty::TyS<'tcx>>,
        seen_requirements: &mut FxHashSet<DefId>,
    ) where
        T: fmt::Display,
    {
        let tcx = self.tcx;
        match *cause_code {
            ObligationCauseCode::ExprAssignable
            | ObligationCauseCode::MatchExpressionArm { .. }
            | ObligationCauseCode::Pattern { .. }
            | ObligationCauseCode::IfExpression { .. }
            | ObligationCauseCode::IfExpressionWithNoElse
            | ObligationCauseCode::MainFunctionType
            | ObligationCauseCode::StartFunctionType
            | ObligationCauseCode::IntrinsicType
            | ObligationCauseCode::MethodReceiver
            | ObligationCauseCode::ReturnNoExpression
            | ObligationCauseCode::UnifyReceiver(..)
            | ObligationCauseCode::MiscObligation => {}
            ObligationCauseCode::SliceOrArrayElem => {
                err.note("slice and array elements must have `Sized` type");
            }
            ObligationCauseCode::TupleElem => {
                err.note("only the last element of a tuple may have a dynamically sized type");
            }
            ObligationCauseCode::ProjectionWf(data) => {
                err.note(&format!("required so that the projection `{}` is well-formed", data,));
            }
            ObligationCauseCode::ReferenceOutlivesReferent(ref_ty) => {
                err.note(&format!(
                    "required so that reference `{}` does not outlive its referent",
                    ref_ty,
                ));
            }
            ObligationCauseCode::ObjectTypeBound(object_ty, region) => {
                err.note(&format!(
                    "required so that the lifetime bound of `{}` for `{}` is satisfied",
                    region, object_ty,
                ));
            }
            ObligationCauseCode::ItemObligation(item_def_id) => {
                let item_name = tcx.def_path_str(item_def_id);
                let msg = format!("required by `{}`", item_name);
                if let Some(sp) = tcx.hir().span_if_local(item_def_id) {
                    let sp = tcx.sess.source_map().guess_head_span(sp);
                    err.span_label(sp, &msg);
                } else {
                    err.note(&msg);
                }
            }
            ObligationCauseCode::BindingObligation(item_def_id, span) => {
                let item_name = tcx.def_path_str(item_def_id);
                let msg = format!("required by this bound in `{}`", item_name);
                if let Some(ident) = tcx.opt_item_name(item_def_id) {
                    let sm = tcx.sess.source_map();
                    let same_line =
                        match (sm.lookup_line(ident.span.hi()), sm.lookup_line(span.lo())) {
                            (Ok(l), Ok(r)) => l.line == r.line,
                            _ => true,
                        };
                    if !ident.span.overlaps(span) && !same_line {
                        err.span_label(ident.span, "required by a bound in this");
                    }
                }
                if span != DUMMY_SP {
                    err.span_label(span, &msg);
                } else {
                    err.note(&msg);
                }
            }
            ObligationCauseCode::ObjectCastObligation(object_ty) => {
                err.note(&format!(
                    "required for the cast to the object type `{}`",
                    self.ty_to_string(object_ty)
                ));
            }
            ObligationCauseCode::Coercion { source: _, target } => {
                err.note(&format!("required by cast to type `{}`", self.ty_to_string(target)));
            }
            ObligationCauseCode::RepeatVec(suggest_const_in_array_repeat_expressions) => {
                err.note(
                    "the `Copy` trait is required because the repeated element will be copied",
                );
                if suggest_const_in_array_repeat_expressions {
                    err.note(
                        "this array initializer can be evaluated at compile-time, see issue \
                         #49147 <https://github.com/rust-lang/rust/issues/49147> \
                         for more information",
                    );
                    if tcx.sess.opts.unstable_features.is_nightly_build() {
                        err.help(
                            "add `#![feature(const_in_array_repeat_expressions)]` to the \
                             crate attributes to enable",
                        );
                    }
                }
            }
            ObligationCauseCode::VariableType(hir_id) => {
                let parent_node = self.tcx.hir().get_parent_node(hir_id);
                match self.tcx.hir().find(parent_node) {
                    Some(Node::Local(hir::Local {
                        init: Some(hir::Expr { kind: hir::ExprKind::Index(_, _), span, .. }),
                        ..
                    })) => {
                        // When encountering an assignment of an unsized trait, like
                        // `let x = ""[..];`, provide a suggestion to borrow the initializer in
                        // order to use have a slice instead.
                        err.span_suggestion_verbose(
                            span.shrink_to_lo(),
                            "consider borrowing here",
                            "&".to_owned(),
                            Applicability::MachineApplicable,
                        );
                        err.note("all local variables must have a statically known size");
                    }
                    Some(Node::Param(param)) => {
                        err.span_suggestion_verbose(
                            param.ty_span.shrink_to_lo(),
                            "function arguments must have a statically known size, borrowed types \
                            always have a known size",
                            "&".to_owned(),
                            Applicability::MachineApplicable,
                        );
                    }
                    _ => {
                        err.note("all local variables must have a statically known size");
                    }
                }
                if !self.tcx.features().unsized_locals {
                    err.help("unsized locals are gated as an unstable feature");
                }
            }
            ObligationCauseCode::SizedArgumentType(sp) => {
                if let Some(span) = sp {
                    err.span_suggestion_verbose(
                        span.shrink_to_lo(),
                        "function arguments must have a statically known size, borrowed types \
                         always have a known size",
                        "&".to_string(),
                        Applicability::MachineApplicable,
                    );
                } else {
                    err.note("all function arguments must have a statically known size");
                }
                if tcx.sess.opts.unstable_features.is_nightly_build()
                    && !self.tcx.features().unsized_fn_params
                {
                    err.help("unsized fn params are gated as an unstable feature");
                }
            }
            ObligationCauseCode::SizedReturnType => {
                err.note("the return type of a function must have a statically known size");
            }
            ObligationCauseCode::SizedYieldType => {
                err.note("the yield type of a generator must have a statically known size");
            }
            ObligationCauseCode::AssignmentLhsSized => {
                err.note("the left-hand-side of an assignment must have a statically known size");
            }
            ObligationCauseCode::TupleInitializerSized => {
                err.note("tuples must have a statically known size to be initialized");
            }
            ObligationCauseCode::StructInitializerSized => {
                err.note("structs must have a statically known size to be initialized");
            }
            ObligationCauseCode::FieldSized { adt_kind: ref item, last, span } => {
                match *item {
                    AdtKind::Struct => {
                        if last {
                            err.note(
                                "the last field of a packed struct may only have a \
                                dynamically sized type if it does not need drop to be run",
                            );
                        } else {
                            err.note(
                                "only the last field of a struct may have a dynamically sized type",
                            );
                        }
                    }
                    AdtKind::Union => {
                        err.note("no field of a union may have a dynamically sized type");
                    }
                    AdtKind::Enum => {
                        err.note("no field of an enum variant may have a dynamically sized type");
                    }
                }
                err.help("change the field's type to have a statically known size");
                err.span_suggestion(
                    span.shrink_to_lo(),
                    "borrowed types always have a statically known size",
                    "&".to_string(),
                    Applicability::MachineApplicable,
                );
                err.multipart_suggestion(
                    "the `Box` type always has a statically known size and allocates its contents \
                     in the heap",
                    vec![
                        (span.shrink_to_lo(), "Box<".to_string()),
                        (span.shrink_to_hi(), ">".to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
            }
            ObligationCauseCode::ConstSized => {
                err.note("constant expressions must have a statically known size");
            }
            ObligationCauseCode::InlineAsmSized => {
                err.note("all inline asm arguments must have a statically known size");
            }
            ObligationCauseCode::ConstPatternStructural => {
                err.note("constants used for pattern-matching must derive `PartialEq` and `Eq`");
            }
            ObligationCauseCode::SharedStatic => {
                err.note("shared static variables must have a type that implements `Sync`");
            }
            ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_ref);
                let ty = parent_trait_ref.skip_binder().self_ty();
                if parent_trait_ref.references_error() {
                    err.cancel();
                    return;
                }

                // If the obligation for a tuple is set directly by a Generator or Closure,
                // then the tuple must be the one containing capture types.
                let is_upvar_tys_infer_tuple = if !matches!(ty.kind(), ty::Tuple(..)) {
                    false
                } else {
                    if let ObligationCauseCode::BuiltinDerivedObligation(ref data) =
                        *data.parent_code
                    {
                        let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_ref);
                        let ty = parent_trait_ref.skip_binder().self_ty();
                        matches!(ty.kind(), ty::Generator(..))
                            || matches!(ty.kind(), ty::Closure(..))
                    } else {
                        false
                    }
                };

                // Don't print the tuple of capture types
                if !is_upvar_tys_infer_tuple {
                    err.note(&format!("required because it appears within the type `{}`", ty));
                }

                obligated_types.push(ty);

                let parent_predicate = parent_trait_ref.without_const().to_predicate(tcx);
                if !self.is_recursive_obligation(obligated_types, &data.parent_code) {
                    // #74711: avoid a stack overflow
                    ensure_sufficient_stack(|| {
                        self.note_obligation_cause_code(
                            err,
                            &parent_predicate,
                            &data.parent_code,
                            obligated_types,
                            seen_requirements,
                        )
                    });
                }
            }
            ObligationCauseCode::ImplDerivedObligation(ref data) => {
                let mut parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_ref);
                let parent_def_id = parent_trait_ref.def_id();
                err.note(&format!(
                    "required because of the requirements on the impl of `{}` for `{}`",
                    parent_trait_ref.print_only_trait_path(),
                    parent_trait_ref.skip_binder().self_ty()
                ));

                let mut parent_predicate = parent_trait_ref.without_const().to_predicate(tcx);
                let mut data = data;
                let mut count = 0;
                seen_requirements.insert(parent_def_id);
                while let ObligationCauseCode::ImplDerivedObligation(child) = &*data.parent_code {
                    // Skip redundant recursive obligation notes. See `ui/issue-20413.rs`.
                    let child_trait_ref = self.resolve_vars_if_possible(child.parent_trait_ref);
                    let child_def_id = child_trait_ref.def_id();
                    if seen_requirements.insert(child_def_id) {
                        break;
                    }
                    count += 1;
                    data = child;
                    parent_predicate = child_trait_ref.without_const().to_predicate(tcx);
                    parent_trait_ref = child_trait_ref;
                }
                if count > 0 {
                    err.note(&format!("{} redundant requirements hidden", count));
                    err.note(&format!(
                        "required because of the requirements on the impl of `{}` for `{}`",
                        parent_trait_ref.print_only_trait_path(),
                        parent_trait_ref.skip_binder().self_ty()
                    ));
                }
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        err,
                        &parent_predicate,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::DerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_ref);
                let parent_predicate = parent_trait_ref.without_const().to_predicate(tcx);
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        err,
                        &parent_predicate,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::CompareImplMethodObligation { .. } => {
                err.note(&format!(
                    "the requirement `{}` appears on the impl method but not on the corresponding \
                     trait method",
                    predicate
                ));
            }
            ObligationCauseCode::CompareImplTypeObligation { .. } => {
                err.note(&format!(
                    "the requirement `{}` appears on the associated impl type but not on the \
                     corresponding associated trait type",
                    predicate
                ));
            }
            ObligationCauseCode::CompareImplConstObligation => {
                err.note(&format!(
                    "the requirement `{}` appears on the associated impl constant \
                     but not on the corresponding associated trait constant",
                    predicate
                ));
            }
            ObligationCauseCode::ReturnType
            | ObligationCauseCode::ReturnValue(_)
            | ObligationCauseCode::BlockTailExpression(_) => (),
            ObligationCauseCode::TrivialBound => {
                err.help("see issue #48214");
                if tcx.sess.opts.unstable_features.is_nightly_build() {
                    err.help("add `#![feature(trivial_bounds)]` to the crate attributes to enable");
                }
            }
        }
    }

    fn suggest_new_overflow_limit(&self, err: &mut DiagnosticBuilder<'_>) {
        let current_limit = self.tcx.sess.recursion_limit();
        let suggested_limit = current_limit * 2;
        err.help(&format!(
            "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate (`{}`)",
            suggested_limit, self.tcx.crate_name,
        ));
    }

    fn suggest_await_before_try(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
        trait_ref: ty::Binder<ty::TraitRef<'tcx>>,
        span: Span,
    ) {
        debug!(
            "suggest_await_before_try: obligation={:?}, span={:?}, trait_ref={:?}, trait_ref_self_ty={:?}",
            obligation,
            span,
            trait_ref,
            trait_ref.self_ty()
        );
        let body_hir_id = obligation.cause.body_id;
        let item_id = self.tcx.hir().get_parent_node(body_hir_id);

        if let Some(body_id) = self.tcx.hir().maybe_body_owned_by(item_id) {
            let body = self.tcx.hir().body(body_id);
            if let Some(hir::GeneratorKind::Async(_)) = body.generator_kind {
                let future_trait = self.tcx.require_lang_item(LangItem::Future, None);

                let self_ty = self.resolve_vars_if_possible(trait_ref.self_ty());

                // Do not check on infer_types to avoid panic in evaluate_obligation.
                if self_ty.has_infer_types() {
                    return;
                }
                let self_ty = self.tcx.erase_regions(self_ty);

                let impls_future = self.tcx.type_implements_trait((
                    future_trait,
                    self_ty.skip_binder(),
                    ty::List::empty(),
                    obligation.param_env,
                ));

                let item_def_id = self
                    .tcx
                    .associated_items(future_trait)
                    .in_definition_order()
                    .next()
                    .unwrap()
                    .def_id;
                // `<T as Future>::Output`
                let projection_ty = ty::ProjectionTy {
                    // `T`
                    substs: self.tcx.mk_substs_trait(
                        trait_ref.self_ty().skip_binder(),
                        self.fresh_substs_for_item(span, item_def_id),
                    ),
                    // `Future::Output`
                    item_def_id,
                };

                let mut selcx = SelectionContext::new(self);

                let mut obligations = vec![];
                let normalized_ty = normalize_projection_type(
                    &mut selcx,
                    obligation.param_env,
                    projection_ty,
                    obligation.cause.clone(),
                    0,
                    &mut obligations,
                );

                debug!(
                    "suggest_await_before_try: normalized_projection_type {:?}",
                    self.resolve_vars_if_possible(normalized_ty)
                );
                let try_obligation = self.mk_trait_obligation_with_new_self_ty(
                    obligation.param_env,
                    trait_ref,
                    normalized_ty,
                );
                debug!("suggest_await_before_try: try_trait_obligation {:?}", try_obligation);
                if self.predicate_may_hold(&try_obligation) && impls_future {
                    if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                        if snippet.ends_with('?') {
                            err.span_suggestion_verbose(
                                span.with_hi(span.hi() - BytePos(1)).shrink_to_hi(),
                                "consider `await`ing on the `Future`",
                                ".await".to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Collect all the returned expressions within the input expression.
/// Used to point at the return spans when we want to suggest some change to them.
#[derive(Default)]
pub struct ReturnsVisitor<'v> {
    pub returns: Vec<&'v hir::Expr<'v>>,
    in_block_tail: bool,
}

impl<'v> Visitor<'v> for ReturnsVisitor<'v> {
    type Map = hir::intravisit::ErasedMap<'v>;

    fn nested_visit_map(&mut self) -> hir::intravisit::NestedVisitorMap<Self::Map> {
        hir::intravisit::NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        // Visit every expression to detect `return` paths, either through the function's tail
        // expression or `return` statements. We walk all nodes to find `return` statements, but
        // we only care about tail expressions when `in_block_tail` is `true`, which means that
        // they're in the return path of the function body.
        match ex.kind {
            hir::ExprKind::Ret(Some(ex)) => {
                self.returns.push(ex);
            }
            hir::ExprKind::Block(block, _) if self.in_block_tail => {
                self.in_block_tail = false;
                for stmt in block.stmts {
                    hir::intravisit::walk_stmt(self, stmt);
                }
                self.in_block_tail = true;
                if let Some(expr) = block.expr {
                    self.visit_expr(expr);
                }
            }
            hir::ExprKind::Match(_, arms, _) if self.in_block_tail => {
                for arm in arms {
                    self.visit_expr(arm.body);
                }
            }
            // We need to walk to find `return`s in the entire body.
            _ if !self.in_block_tail => hir::intravisit::walk_expr(self, ex),
            _ => self.returns.push(ex),
        }
    }

    fn visit_body(&mut self, body: &'v hir::Body<'v>) {
        assert!(!self.in_block_tail);
        if body.generator_kind().is_none() {
            if let hir::ExprKind::Block(block, None) = body.value.kind {
                if block.expr.is_some() {
                    self.in_block_tail = true;
                }
            }
        }
        hir::intravisit::walk_body(self, body);
    }
}

/// Collect all the awaited expressions within the input expression.
#[derive(Default)]
struct AwaitsVisitor {
    awaits: Vec<hir::HirId>,
}

impl<'v> Visitor<'v> for AwaitsVisitor {
    type Map = hir::intravisit::ErasedMap<'v>;

    fn nested_visit_map(&mut self) -> hir::intravisit::NestedVisitorMap<Self::Map> {
        hir::intravisit::NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        if let hir::ExprKind::Yield(_, hir::YieldSource::Await { expr: Some(id) }) = ex.kind {
            self.awaits.push(id)
        }
        hir::intravisit::walk_expr(self, ex)
    }
}

pub trait NextTypeParamName {
    fn next_type_param_name(&self, name: Option<&str>) -> String;
}

impl NextTypeParamName for &[hir::GenericParam<'_>] {
    fn next_type_param_name(&self, name: Option<&str>) -> String {
        // This is the list of possible parameter names that we might suggest.
        let name = name.and_then(|n| n.chars().next()).map(|c| c.to_string().to_uppercase());
        let name = name.as_deref();
        let possible_names = [name.unwrap_or("T"), "T", "U", "V", "X", "Y", "Z", "A", "B", "C"];
        let used_names = self
            .iter()
            .filter_map(|p| match p.name {
                hir::ParamName::Plain(ident) => Some(ident.name),
                _ => None,
            })
            .collect::<Vec<_>>();

        possible_names
            .iter()
            .find(|n| !used_names.contains(&Symbol::intern(n)))
            .unwrap_or(&"ParamName")
            .to_string()
    }
}

fn suggest_trait_object_return_type_alternatives(
    err: &mut DiagnosticBuilder<'_>,
    ret_ty: Span,
    trait_obj: &str,
    is_object_safe: bool,
) {
    err.span_suggestion(
        ret_ty,
        "use some type `T` that is `T: Sized` as the return type if all return paths have the \
            same type",
        "T".to_string(),
        Applicability::MaybeIncorrect,
    );
    err.span_suggestion(
        ret_ty,
        &format!(
            "use `impl {}` as the return type if all return paths have the same type but you \
                want to expose only the trait in the signature",
            trait_obj,
        ),
        format!("impl {}", trait_obj),
        Applicability::MaybeIncorrect,
    );
    if is_object_safe {
        err.span_suggestion(
            ret_ty,
            &format!(
                "use a boxed trait object if all return paths implement trait `{}`",
                trait_obj,
            ),
            format!("Box<dyn {}>", trait_obj),
            Applicability::MaybeIncorrect,
        );
    }
}

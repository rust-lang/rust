//! Give useful errors and suggestions to users when an item can't be
//! found or is otherwise invalid.

use crate::errors;
use crate::Expectation;
use crate::FnCtxt;
use rustc_ast::ast::Mutability;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::StashKey;
use rustc_errors::{
    pluralize, struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
    MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::PatKind::Binding;
use rustc_hir::PathSegment;
use rustc_hir::{ExprKind, Node, QPath};
use rustc_infer::infer::{
    type_variable::{TypeVariableOrigin, TypeVariableOriginKind},
    RegionVariableOrigin,
};
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::traits::util::supertraits;
use rustc_middle::ty::fast_reject::DeepRejectCtxt;
use rustc_middle::ty::fast_reject::{simplify_type, TreatParams};
use rustc_middle::ty::print::{with_crate_prefix, with_forced_trimmed_paths};
use rustc_middle::ty::{self, GenericArgKind, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::ty::{IsSuggestable, ToPolyTraitRef};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Symbol;
use rustc_span::{edit_distance, source_map, ExpnKind, FileName, MacroKind, Span};
use rustc_trait_selection::traits::error_reporting::on_unimplemented::OnUnimplementedNote;
use rustc_trait_selection::traits::error_reporting::on_unimplemented::TypeErrCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    FulfillmentError, Obligation, ObligationCause, ObligationCauseCode,
};

use super::probe::{AutorefOrPtrAdjustment, IsSuggestion, Mode, ProbeScope};
use super::{CandidateSource, MethodError, NoMatchData};
use rustc_hir::intravisit::Visitor;
use std::cmp::{self, Ordering};
use std::iter;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn is_fn_ty(&self, ty: Ty<'tcx>, span: Span) -> bool {
        let tcx = self.tcx;
        match ty.kind() {
            // Not all of these (e.g., unsafe fns) implement `FnOnce`,
            // so we look for these beforehand.
            ty::Closure(..) | ty::FnDef(..) | ty::FnPtr(_) => true,
            // If it's not a simple function, look for things which implement `FnOnce`.
            _ => {
                let Some(fn_once) = tcx.lang_items().fn_once_trait() else {
                    return false;
                };

                // This conditional prevents us from asking to call errors and unresolved types.
                // It might seem that we can use `predicate_must_hold_modulo_regions`,
                // but since a Dummy binder is used to fill in the FnOnce trait's arguments,
                // type resolution always gives a "maybe" here.
                if self.autoderef(span, ty).any(|(ty, _)| {
                    info!("check deref {:?} error", ty);
                    matches!(ty.kind(), ty::Error(_) | ty::Infer(_))
                }) {
                    return false;
                }

                self.autoderef(span, ty).any(|(ty, _)| {
                    info!("check deref {:?} impl FnOnce", ty);
                    self.probe(|_| {
                        let trait_ref = tcx.mk_trait_ref(
                            fn_once,
                            [
                                ty,
                                self.next_ty_var(TypeVariableOrigin {
                                    kind: TypeVariableOriginKind::MiscVariable,
                                    span,
                                }),
                            ],
                        );
                        let poly_trait_ref = ty::Binder::dummy(trait_ref);
                        let obligation = Obligation::misc(
                            tcx,
                            span,
                            self.body_id,
                            self.param_env,
                            poly_trait_ref.without_const(),
                        );
                        self.predicate_may_hold(&obligation)
                    })
                })
            }
        }
    }

    fn is_slice_ty(&self, ty: Ty<'tcx>, span: Span) -> bool {
        self.autoderef(span, ty).any(|(ty, _)| matches!(ty.kind(), ty::Slice(..) | ty::Array(..)))
    }

    #[instrument(level = "debug", skip(self))]
    pub fn report_method_error(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        source: SelfSource<'tcx>,
        error: MethodError<'tcx>,
        args: Option<(&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>])>,
        expected: Expectation<'tcx>,
    ) -> Option<DiagnosticBuilder<'_, ErrorGuaranteed>> {
        // Avoid suggestions when we don't know what's going on.
        if rcvr_ty.references_error() {
            return None;
        }

        let sugg_span = if let SelfSource::MethodCall(expr) = source {
            // Given `foo.bar(baz)`, `expr` is `bar`, but we want to point to the whole thing.
            self.tcx.hir().expect_expr(self.tcx.hir().parent_id(expr.hir_id)).span
        } else {
            span
        };

        match error {
            MethodError::NoMatch(mut no_match_data) => {
                return self.report_no_match_method_error(
                    span,
                    rcvr_ty,
                    item_name,
                    source,
                    args,
                    sugg_span,
                    &mut no_match_data,
                    expected,
                );
            }

            MethodError::Ambiguity(mut sources) => {
                let mut err = struct_span_err!(
                    self.sess(),
                    item_name.span,
                    E0034,
                    "multiple applicable items in scope"
                );
                err.span_label(item_name.span, format!("multiple `{}` found", item_name));

                self.note_candidates_on_method_error(
                    rcvr_ty,
                    item_name,
                    args,
                    span,
                    &mut err,
                    &mut sources,
                    Some(sugg_span),
                );
                err.emit();
            }

            MethodError::PrivateMatch(kind, def_id, out_of_scope_traits) => {
                let kind = self.tcx.def_kind_descr(kind, def_id);
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    item_name.span,
                    E0624,
                    "{} `{}` is private",
                    kind,
                    item_name
                );
                err.span_label(item_name.span, &format!("private {}", kind));
                let sp = self
                    .tcx
                    .hir()
                    .span_if_local(def_id)
                    .unwrap_or_else(|| self.tcx.def_span(def_id));
                err.span_label(sp, &format!("private {} defined here", kind));
                self.suggest_valid_traits(&mut err, out_of_scope_traits);
                err.emit();
            }

            MethodError::IllegalSizedBound { candidates, needs_mut, bound_span, self_expr } => {
                let msg = if needs_mut {
                    with_forced_trimmed_paths!(format!(
                        "the `{item_name}` method cannot be invoked on `{rcvr_ty}`"
                    ))
                } else {
                    format!("the `{item_name}` method cannot be invoked on a trait object")
                };
                let mut err = self.sess().struct_span_err(span, &msg);
                if !needs_mut {
                    err.span_label(bound_span, "this has a `Sized` requirement");
                }
                if !candidates.is_empty() {
                    let help = format!(
                        "{an}other candidate{s} {were} found in the following trait{s}, perhaps \
                         add a `use` for {one_of_them}:",
                        an = if candidates.len() == 1 { "an" } else { "" },
                        s = pluralize!(candidates.len()),
                        were = pluralize!("was", candidates.len()),
                        one_of_them = if candidates.len() == 1 { "it" } else { "one_of_them" },
                    );
                    self.suggest_use_candidates(&mut err, help, candidates);
                }
                if let ty::Ref(region, t_type, mutability) = rcvr_ty.kind() {
                    if needs_mut {
                        let trait_type = self.tcx.mk_ref(
                            *region,
                            ty::TypeAndMut { ty: *t_type, mutbl: mutability.invert() },
                        );
                        let msg = format!("you need `{}` instead of `{}`", trait_type, rcvr_ty);
                        let mut kind = &self_expr.kind;
                        while let hir::ExprKind::AddrOf(_, _, expr)
                        | hir::ExprKind::Unary(hir::UnOp::Deref, expr) = kind
                        {
                            kind = &expr.kind;
                        }
                        if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = kind
                            && let hir::def::Res::Local(hir_id) = path.res
                            && let Some(hir::Node::Pat(b)) = self.tcx.hir().find(hir_id)
                            && let Some(hir::Node::Param(p)) = self.tcx.hir().find_parent(b.hir_id)
                            && let Some(node) = self.tcx.hir().find_parent(p.hir_id)
                            && let Some(decl) = node.fn_decl()
                            && let Some(ty) = decl.inputs.iter().find(|ty| ty.span == p.ty_span)
                            && let hir::TyKind::Ref(_, mut_ty) = &ty.kind
                            && let hir::Mutability::Not = mut_ty.mutbl
                        {
                            err.span_suggestion_verbose(
                                mut_ty.ty.span.shrink_to_lo(),
                                &msg,
                                "mut ",
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.help(&msg);
                        }
                    }
                }
                err.emit();
            }

            MethodError::BadReturnType => bug!("no return type expectations but got BadReturnType"),
        }
        None
    }

    fn suggest_missing_writer(
        &self,
        rcvr_ty: Ty<'tcx>,
        args: (&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>]),
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let (ty_str, _ty_file) = self.tcx.short_ty_string(rcvr_ty);
        let mut err =
            struct_span_err!(self.tcx.sess, args.0.span, E0599, "cannot write into `{}`", ty_str);
        err.span_note(
            args.0.span,
            "must implement `io::Write`, `fmt::Write`, or have a `write_fmt` method",
        );
        if let ExprKind::Lit(_) = args.0.kind {
            err.span_help(
                args.0.span.shrink_to_lo(),
                "a writer is needed before this format string",
            );
        };

        err
    }

    pub fn report_no_match_method_error(
        &self,
        mut span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        source: SelfSource<'tcx>,
        args: Option<(&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>])>,
        sugg_span: Span,
        no_match_data: &mut NoMatchData<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Option<DiagnosticBuilder<'_, ErrorGuaranteed>> {
        let mode = no_match_data.mode;
        let tcx = self.tcx;
        let rcvr_ty = self.resolve_vars_if_possible(rcvr_ty);
        let (ty_str, ty_file) = tcx.short_ty_string(rcvr_ty);
        let short_ty_str = with_forced_trimmed_paths!(rcvr_ty.to_string());
        let is_method = mode == Mode::MethodCall;
        let unsatisfied_predicates = &no_match_data.unsatisfied_predicates;
        let similar_candidate = no_match_data.similar_candidate;
        let item_kind = if is_method {
            "method"
        } else if rcvr_ty.is_enum() {
            "variant or associated item"
        } else {
            match (item_name.as_str().chars().next(), rcvr_ty.is_fresh_ty()) {
                (Some(name), false) if name.is_lowercase() => "function or associated item",
                (Some(_), false) => "associated item",
                (Some(_), true) | (None, false) => "variant or associated item",
                (None, true) => "variant",
            }
        };

        // We could pass the file for long types into these two, but it isn't strictly necessary
        // given how targetted they are.
        if self.suggest_wrapping_range_with_parens(
            tcx,
            rcvr_ty,
            source,
            span,
            item_name,
            &short_ty_str,
        ) || self.suggest_constraining_numerical_ty(
            tcx,
            rcvr_ty,
            source,
            span,
            item_kind,
            item_name,
            &short_ty_str,
        ) {
            return None;
        }
        span = item_name.span;

        // Don't show generic arguments when the method can't be found in any implementation (#81576).
        let mut ty_str_reported = ty_str.clone();
        if let ty::Adt(_, generics) = rcvr_ty.kind() {
            if generics.len() > 0 {
                let mut autoderef = self.autoderef(span, rcvr_ty);
                let candidate_found = autoderef.any(|(ty, _)| {
                    if let ty::Adt(adt_def, _) = ty.kind() {
                        self.tcx
                            .inherent_impls(adt_def.did())
                            .iter()
                            .any(|def_id| self.associated_value(*def_id, item_name).is_some())
                    } else {
                        false
                    }
                });
                let has_deref = autoderef.step_count() > 0;
                if !candidate_found && !has_deref && unsatisfied_predicates.is_empty() {
                    if let Some((path_string, _)) = ty_str.split_once('<') {
                        ty_str_reported = path_string.to_string();
                    }
                }
            }
        }

        let is_write = sugg_span.ctxt().outer_expn_data().macro_def_id.map_or(false, |def_id| {
            tcx.is_diagnostic_item(sym::write_macro, def_id)
                || tcx.is_diagnostic_item(sym::writeln_macro, def_id)
        }) && item_name.name == Symbol::intern("write_fmt");
        let mut err = if is_write
            && let Some(args) = args
        {
            self.suggest_missing_writer(rcvr_ty, args)
        } else {
            struct_span_err!(
                tcx.sess,
                span,
                E0599,
                "no {} named `{}` found for {} `{}` in the current scope",
                item_kind,
                item_name,
                rcvr_ty.prefix_string(self.tcx),
                ty_str_reported,
            )
        };
        if tcx.sess.source_map().is_multiline(sugg_span) {
            err.span_label(sugg_span.with_hi(span.lo()), "");
        }
        let ty_str = if short_ty_str.len() < ty_str.len() && ty_str.len() > 10 {
            short_ty_str
        } else {
            ty_str
        };
        if let Some(file) = ty_file {
            err.note(&format!("the full type name has been written to '{}'", file.display(),));
        }
        if rcvr_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        if tcx.ty_is_opaque_future(rcvr_ty) && item_name.name == sym::poll {
            err.help(&format!(
                "method `poll` found on `Pin<&mut {ty_str}>`, \
                see documentation for `std::pin::Pin`"
            ));
            err.help("self type must be pinned to call `Future::poll`, \
                see https://rust-lang.github.io/async-book/04_pinning/01_chapter.html#pinning-in-practice"
            );
        }

        if let Mode::MethodCall = mode && let SelfSource::MethodCall(cal) = source {
            self.suggest_await_before_method(
                &mut err, item_name, rcvr_ty, cal, span, expected.only_has_type(self),
            );
        }
        if let Some(span) =
            tcx.resolutions(()).confused_type_with_std_module.get(&span.with_parent(None))
        {
            err.span_suggestion(
                span.shrink_to_lo(),
                "you are looking for the module in `std`, not the primitive type",
                "std::",
                Applicability::MachineApplicable,
            );
        }
        if let ty::RawPtr(_) = &rcvr_ty.kind() {
            err.note(
                "try using `<*const T>::as_ref()` to get a reference to the \
                 type behind the pointer: https://doc.rust-lang.org/std/\
                 primitive.pointer.html#method.as_ref",
            );
            err.note(
                "using `<*const T>::as_ref()` on a pointer which is unaligned or points \
                 to invalid or uninitialized memory is undefined behavior",
            );
        }

        let ty_span = match rcvr_ty.kind() {
            ty::Param(param_type) => {
                Some(param_type.span_from_generics(self.tcx, self.body_id.to_def_id()))
            }
            ty::Adt(def, _) if def.did().is_local() => Some(tcx.def_span(def.did())),
            _ => None,
        };
        if let Some(span) = ty_span {
            err.span_label(
                span,
                format!(
                    "{item_kind} `{item_name}` not found for this {}",
                    rcvr_ty.prefix_string(self.tcx)
                ),
            );
        }

        if let SelfSource::MethodCall(rcvr_expr) = source {
            self.suggest_fn_call(&mut err, rcvr_expr, rcvr_ty, |output_ty| {
                let call_expr =
                    self.tcx.hir().expect_expr(self.tcx.hir().parent_id(rcvr_expr.hir_id));
                let probe = self.lookup_probe_for_diagnostic(
                    item_name,
                    output_ty,
                    call_expr,
                    ProbeScope::AllTraits,
                    expected.only_has_type(self),
                );
                probe.is_ok()
            });

            self.note_internal_mutation_in_method(
                &mut err,
                rcvr_expr,
                expected.to_option(&self),
                rcvr_ty,
            );
        }

        let mut custom_span_label = false;

        let static_candidates = &mut no_match_data.static_candidates;
        if !static_candidates.is_empty() {
            err.note(
                "found the following associated functions; to be used as methods, \
                 functions must have a `self` parameter",
            );
            err.span_label(span, "this is an associated function, not a method");
            custom_span_label = true;
        }
        if static_candidates.len() == 1 {
            self.suggest_associated_call_syntax(
                &mut err,
                &static_candidates,
                rcvr_ty,
                source,
                item_name,
                args,
                sugg_span,
            );
            self.note_candidates_on_method_error(
                rcvr_ty,
                item_name,
                args,
                span,
                &mut err,
                static_candidates,
                None,
            );
        } else if static_candidates.len() > 1 {
            self.note_candidates_on_method_error(
                rcvr_ty,
                item_name,
                args,
                span,
                &mut err,
                static_candidates,
                Some(sugg_span),
            );
        }

        let mut bound_spans = vec![];
        let mut restrict_type_params = false;
        let mut unsatisfied_bounds = false;
        if item_name.name == sym::count && self.is_slice_ty(rcvr_ty, span) {
            let msg = "consider using `len` instead";
            if let SelfSource::MethodCall(_expr) = source {
                err.span_suggestion_short(span, msg, "len", Applicability::MachineApplicable);
            } else {
                err.span_label(span, msg);
            }
            if let Some(iterator_trait) = self.tcx.get_diagnostic_item(sym::Iterator) {
                let iterator_trait = self.tcx.def_path_str(iterator_trait);
                err.note(&format!(
                    "`count` is defined on `{iterator_trait}`, which `{rcvr_ty}` does not implement"
                ));
            }
        } else if !unsatisfied_predicates.is_empty() {
            let mut type_params = FxHashMap::default();

            // Pick out the list of unimplemented traits on the receiver.
            // This is used for custom error messages with the `#[rustc_on_unimplemented]` attribute.
            let mut unimplemented_traits = FxHashMap::default();
            let mut unimplemented_traits_only = true;
            for (predicate, _parent_pred, cause) in unsatisfied_predicates {
                if let (ty::PredicateKind::Clause(ty::Clause::Trait(p)), Some(cause)) =
                    (predicate.kind().skip_binder(), cause.as_ref())
                {
                    if p.trait_ref.self_ty() != rcvr_ty {
                        // This is necessary, not just to keep the errors clean, but also
                        // because our derived obligations can wind up with a trait ref that
                        // requires a different param_env to be correctly compared.
                        continue;
                    }
                    unimplemented_traits.entry(p.trait_ref.def_id).or_insert((
                        predicate.kind().rebind(p.trait_ref),
                        Obligation {
                            cause: cause.clone(),
                            param_env: self.param_env,
                            predicate: *predicate,
                            recursion_depth: 0,
                        },
                    ));
                }
            }

            // Make sure that, if any traits other than the found ones were involved,
            // we don't don't report an unimplemented trait.
            // We don't want to say that `iter::Cloned` is not an iterator, just
            // because of some non-Clone item being iterated over.
            for (predicate, _parent_pred, _cause) in unsatisfied_predicates {
                match predicate.kind().skip_binder() {
                    ty::PredicateKind::Clause(ty::Clause::Trait(p))
                        if unimplemented_traits.contains_key(&p.trait_ref.def_id) => {}
                    _ => {
                        unimplemented_traits_only = false;
                        break;
                    }
                }
            }

            let mut collect_type_param_suggestions =
                |self_ty: Ty<'tcx>, parent_pred: ty::Predicate<'tcx>, obligation: &str| {
                    // We don't care about regions here, so it's fine to skip the binder here.
                    if let (ty::Param(_), ty::PredicateKind::Clause(ty::Clause::Trait(p))) =
                        (self_ty.kind(), parent_pred.kind().skip_binder())
                    {
                        let hir = self.tcx.hir();
                        let node = match p.trait_ref.self_ty().kind() {
                            ty::Param(_) => {
                                // Account for `fn` items like in `issue-35677.rs` to
                                // suggest restricting its type params.
                                Some(hir.get_by_def_id(self.body_id))
                            }
                            ty::Adt(def, _) => {
                                def.did().as_local().map(|def_id| hir.get_by_def_id(def_id))
                            }
                            _ => None,
                        };
                        if let Some(hir::Node::Item(hir::Item { kind, .. })) = node
                            && let Some(g) = kind.generics()
                        {
                            let key = (
                                g.tail_span_for_predicate_suggestion(),
                                g.add_where_or_trailing_comma(),
                            );
                            type_params
                                .entry(key)
                                .or_insert_with(FxHashSet::default)
                                .insert(obligation.to_owned());
                            return true;
                        }
                    }
                    false
                };
            let mut bound_span_label = |self_ty: Ty<'_>, obligation: &str, quiet: &str| {
                let msg = format!(
                    "doesn't satisfy `{}`",
                    if obligation.len() > 50 { quiet } else { obligation }
                );
                match &self_ty.kind() {
                    // Point at the type that couldn't satisfy the bound.
                    ty::Adt(def, _) => bound_spans.push((self.tcx.def_span(def.did()), msg)),
                    // Point at the trait object that couldn't satisfy the bound.
                    ty::Dynamic(preds, _, _) => {
                        for pred in preds.iter() {
                            match pred.skip_binder() {
                                ty::ExistentialPredicate::Trait(tr) => {
                                    bound_spans.push((self.tcx.def_span(tr.def_id), msg.clone()))
                                }
                                ty::ExistentialPredicate::Projection(_)
                                | ty::ExistentialPredicate::AutoTrait(_) => {}
                            }
                        }
                    }
                    // Point at the closure that couldn't satisfy the bound.
                    ty::Closure(def_id, _) => bound_spans
                        .push((tcx.def_span(*def_id), format!("doesn't satisfy `{}`", quiet))),
                    _ => {}
                }
            };
            let mut format_pred = |pred: ty::Predicate<'tcx>| {
                let bound_predicate = pred.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Clause(ty::Clause::Projection(pred)) => {
                        let pred = bound_predicate.rebind(pred);
                        // `<Foo as Iterator>::Item = String`.
                        let projection_ty = pred.skip_binder().projection_ty;

                        let substs_with_infer_self = tcx.mk_substs_from_iter(
                            iter::once(tcx.mk_ty_var(ty::TyVid::from_u32(0)).into())
                                .chain(projection_ty.substs.iter().skip(1)),
                        );

                        let quiet_projection_ty =
                            tcx.mk_alias_ty(projection_ty.def_id, substs_with_infer_self);

                        let term = pred.skip_binder().term;

                        let obligation = format!("{} = {}", projection_ty, term);
                        let quiet = with_forced_trimmed_paths!(format!(
                            "{} = {}",
                            quiet_projection_ty, term
                        ));

                        bound_span_label(projection_ty.self_ty(), &obligation, &quiet);
                        Some((obligation, projection_ty.self_ty()))
                    }
                    ty::PredicateKind::Clause(ty::Clause::Trait(poly_trait_ref)) => {
                        let p = poly_trait_ref.trait_ref;
                        let self_ty = p.self_ty();
                        let path = p.print_only_trait_path();
                        let obligation = format!("{}: {}", self_ty, path);
                        let quiet = with_forced_trimmed_paths!(format!("_: {}", path));
                        bound_span_label(self_ty, &obligation, &quiet);
                        Some((obligation, self_ty))
                    }
                    _ => None,
                }
            };

            // Find all the requirements that come from a local `impl` block.
            let mut skip_list: FxHashSet<_> = Default::default();
            let mut spanned_predicates = FxHashMap::default();
            for (p, parent_p, impl_def_id, cause) in unsatisfied_predicates
                .iter()
                .filter_map(|(p, parent, c)| c.as_ref().map(|c| (p, parent, c)))
                .filter_map(|(p, parent, c)| match c.code() {
                    ObligationCauseCode::ImplDerivedObligation(data)
                        if matches!(p.kind().skip_binder(), ty::PredicateKind::Clause(_)) =>
                    {
                        Some((p, parent, data.impl_or_alias_def_id, data))
                    }
                    _ => None,
                })
            {
                match self.tcx.hir().get_if_local(impl_def_id) {
                    // Unmet obligation comes from a `derive` macro, point at it once to
                    // avoid multiple span labels pointing at the same place.
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, .. }),
                        ..
                    })) if matches!(
                        self_ty.span.ctxt().outer_expn_data().kind,
                        ExpnKind::Macro(MacroKind::Derive, _)
                    ) || matches!(
                        of_trait.as_ref().map(|t| t.path.span.ctxt().outer_expn_data().kind),
                        Some(ExpnKind::Macro(MacroKind::Derive, _))
                    ) =>
                    {
                        let span = self_ty.span.ctxt().outer_expn_data().call_site;
                        let entry = spanned_predicates.entry(span);
                        let entry = entry.or_insert_with(|| {
                            (FxHashSet::default(), FxHashSet::default(), Vec::new())
                        });
                        entry.0.insert(span);
                        entry.1.insert((
                            span,
                            "unsatisfied trait bound introduced in this `derive` macro",
                        ));
                        entry.2.push(p);
                        skip_list.insert(p);
                    }

                    // Unmet obligation coming from an `impl`.
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, generics, .. }),
                        span: item_span,
                        ..
                    })) => {
                        let sized_pred =
                            unsatisfied_predicates.iter().any(|(pred, _, _)| {
                                match pred.kind().skip_binder() {
                                    ty::PredicateKind::Clause(ty::Clause::Trait(pred)) => {
                                        Some(pred.def_id()) == self.tcx.lang_items().sized_trait()
                                            && pred.polarity == ty::ImplPolarity::Positive
                                    }
                                    _ => false,
                                }
                            });
                        for param in generics.params {
                            if param.span == cause.span && sized_pred {
                                let (sp, sugg) = match param.colon_span {
                                    Some(sp) => (sp.shrink_to_hi(), " ?Sized +"),
                                    None => (param.span.shrink_to_hi(), ": ?Sized"),
                                };
                                err.span_suggestion_verbose(
                                    sp,
                                    "consider relaxing the type parameter's implicit `Sized` bound",
                                    sugg,
                                    Applicability::MachineApplicable,
                                );
                            }
                        }
                        if let Some(pred) = parent_p {
                            // Done to add the "doesn't satisfy" `span_label`.
                            let _ = format_pred(*pred);
                        }
                        skip_list.insert(p);
                        let entry = spanned_predicates.entry(self_ty.span);
                        let entry = entry.or_insert_with(|| {
                            (FxHashSet::default(), FxHashSet::default(), Vec::new())
                        });
                        entry.2.push(p);
                        if cause.span != *item_span {
                            entry.0.insert(cause.span);
                            entry.1.insert((cause.span, "unsatisfied trait bound introduced here"));
                        } else {
                            if let Some(trait_ref) = of_trait {
                                entry.0.insert(trait_ref.path.span);
                            }
                            entry.0.insert(self_ty.span);
                        };
                        if let Some(trait_ref) = of_trait {
                            entry.1.insert((trait_ref.path.span, ""));
                        }
                        entry.1.insert((self_ty.span, ""));
                    }
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Trait(rustc_ast::ast::IsAuto::Yes, ..),
                        span: item_span,
                        ..
                    })) => {
                        tcx.sess.delay_span_bug(
                            *item_span,
                            "auto trait is invoked with no method error, but no error reported?",
                        );
                    }
                    Some(Node::Item(hir::Item {
                        ident,
                        kind: hir::ItemKind::Trait(..) | hir::ItemKind::TraitAlias(..),
                        ..
                    })) => {
                        skip_list.insert(p);
                        let entry = spanned_predicates.entry(ident.span);
                        let entry = entry.or_insert_with(|| {
                            (FxHashSet::default(), FxHashSet::default(), Vec::new())
                        });
                        entry.0.insert(cause.span);
                        entry.1.insert((ident.span, ""));
                        entry.1.insert((cause.span, "unsatisfied trait bound introduced here"));
                        entry.2.push(p);
                    }
                    Some(node) => unreachable!("encountered `{node:?}`"),
                    None => (),
                }
            }
            let mut spanned_predicates: Vec<_> = spanned_predicates.into_iter().collect();
            spanned_predicates.sort_by_key(|(span, _)| *span);
            for (_, (primary_spans, span_labels, predicates)) in spanned_predicates {
                let mut preds: Vec<_> = predicates
                    .iter()
                    .filter_map(|pred| format_pred(**pred))
                    .map(|(p, _)| format!("`{}`", p))
                    .collect();
                preds.sort();
                preds.dedup();
                let msg = if let [pred] = &preds[..] {
                    format!("trait bound {} was not satisfied", pred)
                } else {
                    format!("the following trait bounds were not satisfied:\n{}", preds.join("\n"),)
                };
                let mut span: MultiSpan = primary_spans.into_iter().collect::<Vec<_>>().into();
                for (sp, label) in span_labels {
                    span.push_span_label(sp, label);
                }
                err.span_note(span, &msg);
                unsatisfied_bounds = true;
            }

            let mut suggested_bounds = FxHashSet::default();
            // The requirements that didn't have an `impl` span to show.
            let mut bound_list = unsatisfied_predicates
                .iter()
                .filter_map(|(pred, parent_pred, _cause)| {
                    let mut suggested = false;
                    format_pred(*pred).map(|(p, self_ty)| {
                        if let Some(parent) = parent_pred && suggested_bounds.contains(parent) {
                            // We don't suggest `PartialEq` when we already suggest `Eq`.
                        } else if !suggested_bounds.contains(pred) {
                            if collect_type_param_suggestions(self_ty, *pred, &p) {
                                suggested = true;
                                suggested_bounds.insert(pred);
                            }
                        }
                        (
                            match parent_pred {
                                None => format!("`{}`", &p),
                                Some(parent_pred) => match format_pred(*parent_pred) {
                                    None => format!("`{}`", &p),
                                    Some((parent_p, _)) => {
                                        if !suggested
                                            && !suggested_bounds.contains(pred)
                                            && !suggested_bounds.contains(parent_pred)
                                        {
                                            if collect_type_param_suggestions(
                                                self_ty,
                                                *parent_pred,
                                                &p,
                                            ) {
                                                suggested_bounds.insert(pred);
                                            }
                                        }
                                        format!("`{}`\nwhich is required by `{}`", p, parent_p)
                                    }
                                },
                            },
                            *pred,
                        )
                    })
                })
                .filter(|(_, pred)| !skip_list.contains(&pred))
                .map(|(t, _)| t)
                .enumerate()
                .collect::<Vec<(usize, String)>>();

            for ((span, add_where_or_comma), obligations) in type_params.into_iter() {
                restrict_type_params = true;
                // #74886: Sort here so that the output is always the same.
                let mut obligations = obligations.into_iter().collect::<Vec<_>>();
                obligations.sort();
                err.span_suggestion_verbose(
                    span,
                    &format!(
                        "consider restricting the type parameter{s} to satisfy the \
                         trait bound{s}",
                        s = pluralize!(obligations.len())
                    ),
                    format!("{} {}", add_where_or_comma, obligations.join(", ")),
                    Applicability::MaybeIncorrect,
                );
            }

            bound_list.sort_by(|(_, a), (_, b)| a.cmp(b)); // Sort alphabetically.
            bound_list.dedup_by(|(_, a), (_, b)| a == b); // #35677
            bound_list.sort_by_key(|(pos, _)| *pos); // Keep the original predicate order.

            if !bound_list.is_empty() || !skip_list.is_empty() {
                let bound_list =
                    bound_list.into_iter().map(|(_, path)| path).collect::<Vec<_>>().join("\n");
                let actual_prefix = rcvr_ty.prefix_string(self.tcx);
                info!("unimplemented_traits.len() == {}", unimplemented_traits.len());
                let (primary_message, label) = if unimplemented_traits.len() == 1
                    && unimplemented_traits_only
                {
                    unimplemented_traits
                        .into_iter()
                        .next()
                        .map(|(_, (trait_ref, obligation))| {
                            if trait_ref.self_ty().references_error() || rcvr_ty.references_error()
                            {
                                // Avoid crashing.
                                return (None, None);
                            }
                            let OnUnimplementedNote { message, label, .. } =
                                self.err_ctxt().on_unimplemented_note(trait_ref, &obligation);
                            (message, label)
                        })
                        .unwrap()
                } else {
                    (None, None)
                };
                let primary_message = primary_message.unwrap_or_else(|| {
                    format!(
                        "the {item_kind} `{item_name}` exists for {actual_prefix} `{ty_str}`, \
                         but its trait bounds were not satisfied"
                    )
                });
                err.set_primary_message(&primary_message);
                if let Some(label) = label {
                    custom_span_label = true;
                    err.span_label(span, label);
                }
                if !bound_list.is_empty() {
                    err.note(&format!(
                        "the following trait bounds were not satisfied:\n{bound_list}"
                    ));
                }
                self.suggest_derive(&mut err, &unsatisfied_predicates);

                unsatisfied_bounds = true;
            }
        }

        let label_span_not_found = |err: &mut Diagnostic| {
            if unsatisfied_predicates.is_empty() {
                err.span_label(span, format!("{item_kind} not found in `{ty_str}`"));
                let is_string_or_ref_str = match rcvr_ty.kind() {
                    ty::Ref(_, ty, _) => {
                        ty.is_str()
                            || matches!(
                                ty.kind(),
                                ty::Adt(adt, _) if Some(adt.did()) == self.tcx.lang_items().string()
                            )
                    }
                    ty::Adt(adt, _) => Some(adt.did()) == self.tcx.lang_items().string(),
                    _ => false,
                };
                if is_string_or_ref_str && item_name.name == sym::iter {
                    err.span_suggestion_verbose(
                        item_name.span,
                        "because of the in-memory representation of `&str`, to obtain \
                         an `Iterator` over each of its codepoint use method `chars`",
                        "chars",
                        Applicability::MachineApplicable,
                    );
                }
                if let ty::Adt(adt, _) = rcvr_ty.kind() {
                    let mut inherent_impls_candidate = self
                        .tcx
                        .inherent_impls(adt.did())
                        .iter()
                        .copied()
                        .filter(|def_id| {
                            if let Some(assoc) = self.associated_value(*def_id, item_name) {
                                // Check for both mode is the same so we avoid suggesting
                                // incorrect associated item.
                                match (mode, assoc.fn_has_self_parameter, source) {
                                    (Mode::MethodCall, true, SelfSource::MethodCall(_)) => {
                                        // We check that the suggest type is actually
                                        // different from the received one
                                        // So we avoid suggestion method with Box<Self>
                                        // for instance
                                        self.tcx.at(span).type_of(*def_id).subst_identity()
                                            != rcvr_ty
                                            && self.tcx.at(span).type_of(*def_id).subst_identity()
                                                != rcvr_ty
                                    }
                                    (Mode::Path, false, _) => true,
                                    _ => false,
                                }
                            } else {
                                false
                            }
                        })
                        .collect::<Vec<_>>();
                    if !inherent_impls_candidate.is_empty() {
                        inherent_impls_candidate.sort();
                        inherent_impls_candidate.dedup();

                        // number of type to shows at most.
                        let limit = if inherent_impls_candidate.len() == 5 { 5 } else { 4 };
                        let type_candidates = inherent_impls_candidate
                            .iter()
                            .take(limit)
                            .map(|impl_item| {
                                format!(
                                    "- `{}`",
                                    self.tcx.at(span).type_of(*impl_item).subst_identity()
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        let additional_types = if inherent_impls_candidate.len() > limit {
                            format!("\nand {} more types", inherent_impls_candidate.len() - limit)
                        } else {
                            "".to_string()
                        };
                        err.note(&format!(
                            "the {item_kind} was found for\n{}{}",
                            type_candidates, additional_types
                        ));
                    }
                }
            } else {
                let ty_str =
                    if ty_str.len() > 50 { String::new() } else { format!("on `{ty_str}` ") };
                err.span_label(
                    span,
                    format!("{item_kind} cannot be called {ty_str}due to unsatisfied trait bounds"),
                );
            }
        };

        // If the method name is the name of a field with a function or closure type,
        // give a helping note that it has to be called as `(x.f)(...)`.
        if let SelfSource::MethodCall(expr) = source {
            if !self.suggest_calling_field_as_fn(span, rcvr_ty, expr, item_name, &mut err)
                && similar_candidate.is_none()
                && !custom_span_label
            {
                label_span_not_found(&mut err);
            }
        } else if !custom_span_label {
            label_span_not_found(&mut err);
        }

        // Don't suggest (for example) `expr.field.clone()` if `expr.clone()`
        // can't be called due to `typeof(expr): Clone` not holding.
        if unsatisfied_predicates.is_empty() {
            self.suggest_calling_method_on_field(
                &mut err,
                source,
                span,
                rcvr_ty,
                item_name,
                expected.only_has_type(self),
            );
        }

        self.check_for_inner_self(&mut err, source, rcvr_ty, item_name);

        bound_spans.sort();
        bound_spans.dedup();
        for (span, msg) in bound_spans.into_iter() {
            err.span_label(span, &msg);
        }

        if rcvr_ty.is_numeric() && rcvr_ty.is_fresh() || restrict_type_params {
        } else {
            self.suggest_traits_to_import(
                &mut err,
                span,
                rcvr_ty,
                item_name,
                args.map(|(_, args)| args.len() + 1),
                source,
                no_match_data.out_of_scope_traits.clone(),
                &unsatisfied_predicates,
                &static_candidates,
                unsatisfied_bounds,
                expected.only_has_type(self),
            );
        }

        // Don't emit a suggestion if we found an actual method
        // that had unsatisfied trait bounds
        if unsatisfied_predicates.is_empty() && rcvr_ty.is_enum() {
            let adt_def = rcvr_ty.ty_adt_def().expect("enum is not an ADT");
            if let Some(suggestion) = edit_distance::find_best_match_for_name(
                &adt_def.variants().iter().map(|s| s.name).collect::<Vec<_>>(),
                item_name.name,
                None,
            ) {
                err.span_suggestion(
                    span,
                    "there is a variant with a similar name",
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
        }

        if item_name.name == sym::as_str && rcvr_ty.peel_refs().is_str() {
            let msg = "remove this method call";
            let mut fallback_span = true;
            if let SelfSource::MethodCall(expr) = source {
                let call_expr = self.tcx.hir().expect_expr(self.tcx.hir().parent_id(expr.hir_id));
                if let Some(span) = call_expr.span.trim_start(expr.span) {
                    err.span_suggestion(span, msg, "", Applicability::MachineApplicable);
                    fallback_span = false;
                }
            }
            if fallback_span {
                err.span_label(span, msg);
            }
        } else if let Some(similar_candidate) = similar_candidate {
            // Don't emit a suggestion if we found an actual method
            // that had unsatisfied trait bounds
            if unsatisfied_predicates.is_empty() {
                let def_kind = similar_candidate.kind.as_def_kind();
                // Methods are defined within the context of a struct and their first parameter is always self,
                // which represents the instance of the struct the method is being called on
                // Associated functions don’t take self as a parameter and
                // they are not methods because they don’t have an instance of the struct to work with.
                if def_kind == DefKind::AssocFn && similar_candidate.fn_has_self_parameter {
                    err.span_suggestion(
                        span,
                        "there is a method with a similar name",
                        similar_candidate.name,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    err.span_suggestion(
                        span,
                        &format!(
                            "there is {} {} with a similar name",
                            self.tcx.def_kind_descr_article(def_kind, similar_candidate.def_id),
                            self.tcx.def_kind_descr(def_kind, similar_candidate.def_id)
                        ),
                        similar_candidate.name,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }

        self.check_for_deref_method(&mut err, source, rcvr_ty, item_name, expected);
        return Some(err);
    }

    fn note_candidates_on_method_error(
        &self,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        args: Option<(&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>])>,
        span: Span,
        err: &mut Diagnostic,
        sources: &mut Vec<CandidateSource>,
        sugg_span: Option<Span>,
    ) {
        sources.sort();
        sources.dedup();
        // Dynamic limit to avoid hiding just one candidate, which is silly.
        let limit = if sources.len() == 5 { 5 } else { 4 };

        for (idx, source) in sources.iter().take(limit).enumerate() {
            match *source {
                CandidateSource::Impl(impl_did) => {
                    // Provide the best span we can. Use the item, if local to crate, else
                    // the impl, if local to crate (item may be defaulted), else nothing.
                    let Some(item) = self.associated_value(impl_did, item_name).or_else(|| {
                        let impl_trait_ref = self.tcx.impl_trait_ref(impl_did)?;
                        self.associated_value(impl_trait_ref.skip_binder().def_id, item_name)
                    }) else {
                        continue;
                    };

                    let note_span = if item.def_id.is_local() {
                        Some(self.tcx.def_span(item.def_id))
                    } else if impl_did.is_local() {
                        Some(self.tcx.def_span(impl_did))
                    } else {
                        None
                    };

                    let impl_ty = self.tcx.at(span).type_of(impl_did).subst_identity();

                    let insertion = match self.tcx.impl_trait_ref(impl_did) {
                        None => String::new(),
                        Some(trait_ref) => {
                            format!(
                                " of the trait `{}`",
                                self.tcx.def_path_str(trait_ref.skip_binder().def_id)
                            )
                        }
                    };

                    let (note_str, idx) = if sources.len() > 1 {
                        (
                            format!(
                                "candidate #{} is defined in an impl{} for the type `{}`",
                                idx + 1,
                                insertion,
                                impl_ty,
                            ),
                            Some(idx + 1),
                        )
                    } else {
                        (
                            format!(
                                "the candidate is defined in an impl{} for the type `{}`",
                                insertion, impl_ty,
                            ),
                            None,
                        )
                    };
                    if let Some(note_span) = note_span {
                        // We have a span pointing to the method. Show note with snippet.
                        err.span_note(note_span, &note_str);
                    } else {
                        err.note(&note_str);
                    }
                    if let Some(sugg_span) = sugg_span
                        && let Some(trait_ref) = self.tcx.impl_trait_ref(impl_did) {
                        let path = self.tcx.def_path_str(trait_ref.skip_binder().def_id);

                        let ty = match item.kind {
                            ty::AssocKind::Const | ty::AssocKind::Type => rcvr_ty,
                            ty::AssocKind::Fn => self
                                .tcx
                                .fn_sig(item.def_id)
                                .subst_identity()
                                .inputs()
                                .skip_binder()
                                .get(0)
                                .filter(|ty| ty.is_ref() && !rcvr_ty.is_ref())
                                .copied()
                                .unwrap_or(rcvr_ty),
                        };
                        print_disambiguation_help(
                            item_name,
                            args,
                            err,
                            path,
                            ty,
                            item.kind,
                            self.tcx.def_kind_descr(item.kind.as_def_kind(), item.def_id),
                            sugg_span,
                            idx,
                            self.tcx.sess.source_map(),
                            item.fn_has_self_parameter,
                        );
                    }
                }
                CandidateSource::Trait(trait_did) => {
                    let Some(item) = self.associated_value(trait_did, item_name) else { continue };
                    let item_span = self.tcx.def_span(item.def_id);
                    let idx = if sources.len() > 1 {
                        let msg = &format!(
                            "candidate #{} is defined in the trait `{}`",
                            idx + 1,
                            self.tcx.def_path_str(trait_did)
                        );
                        err.span_note(item_span, msg);
                        Some(idx + 1)
                    } else {
                        let msg = &format!(
                            "the candidate is defined in the trait `{}`",
                            self.tcx.def_path_str(trait_did)
                        );
                        err.span_note(item_span, msg);
                        None
                    };
                    if let Some(sugg_span) = sugg_span {
                        let path = self.tcx.def_path_str(trait_did);
                        print_disambiguation_help(
                            item_name,
                            args,
                            err,
                            path,
                            rcvr_ty,
                            item.kind,
                            self.tcx.def_kind_descr(item.kind.as_def_kind(), item.def_id),
                            sugg_span,
                            idx,
                            self.tcx.sess.source_map(),
                            item.fn_has_self_parameter,
                        );
                    }
                }
            }
        }
        if sources.len() > limit {
            err.note(&format!("and {} others", sources.len() - limit));
        }
    }

    /// Suggest calling `Ty::method` if `.method()` isn't found because the method
    /// doesn't take a `self` receiver.
    fn suggest_associated_call_syntax(
        &self,
        err: &mut Diagnostic,
        static_candidates: &Vec<CandidateSource>,
        rcvr_ty: Ty<'tcx>,
        source: SelfSource<'tcx>,
        item_name: Ident,
        args: Option<(&hir::Expr<'tcx>, &[hir::Expr<'tcx>])>,
        sugg_span: Span,
    ) {
        let mut has_unsuggestable_args = false;
        let ty_str = if let Some(CandidateSource::Impl(impl_did)) = static_candidates.get(0) {
            // When the "method" is resolved through dereferencing, we really want the
            // original type that has the associated function for accurate suggestions.
            // (#61411)
            let impl_ty = self.tcx.type_of(*impl_did).subst_identity();
            let target_ty = self
                .autoderef(sugg_span, rcvr_ty)
                .find(|(rcvr_ty, _)| {
                    DeepRejectCtxt { treat_obligation_params: TreatParams::AsCandidateKey }
                        .types_may_unify(*rcvr_ty, impl_ty)
                })
                .map_or(impl_ty, |(ty, _)| ty)
                .peel_refs();
            if let ty::Adt(def, substs) = target_ty.kind() {
                // If there are any inferred arguments, (`{integer}`), we should replace
                // them with underscores to allow the compiler to infer them
                let infer_substs = self.tcx.mk_substs_from_iter(substs.into_iter().map(|arg| {
                    if !arg.is_suggestable(self.tcx, true) {
                        has_unsuggestable_args = true;
                        match arg.unpack() {
                            GenericArgKind::Lifetime(_) => self
                                .next_region_var(RegionVariableOrigin::MiscVariable(
                                    rustc_span::DUMMY_SP,
                                ))
                                .into(),
                            GenericArgKind::Type(_) => self
                                .next_ty_var(TypeVariableOrigin {
                                    span: rustc_span::DUMMY_SP,
                                    kind: TypeVariableOriginKind::MiscVariable,
                                })
                                .into(),
                            GenericArgKind::Const(arg) => self
                                .next_const_var(
                                    arg.ty(),
                                    ConstVariableOrigin {
                                        span: rustc_span::DUMMY_SP,
                                        kind: ConstVariableOriginKind::MiscVariable,
                                    },
                                )
                                .into(),
                        }
                    } else {
                        arg
                    }
                }));

                self.tcx.value_path_str_with_substs(def.did(), infer_substs)
            } else {
                self.ty_to_value_string(target_ty)
            }
        } else {
            self.ty_to_value_string(rcvr_ty.peel_refs())
        };
        if let SelfSource::MethodCall(_) = source {
            let first_arg = if let Some(CandidateSource::Impl(impl_did)) = static_candidates.get(0)
                && let Some(assoc) = self.associated_value(*impl_did, item_name)
                && assoc.kind == ty::AssocKind::Fn
            {
                let sig = self.tcx.fn_sig(assoc.def_id).subst_identity();
                sig.inputs().skip_binder().get(0).and_then(|first| if first.peel_refs() == rcvr_ty.peel_refs() {
                    None
                } else {
                    Some(first.ref_mutability().map_or("", |mutbl| mutbl.ref_prefix_str()))
                })
            } else {
                None
            };
            let mut applicability = Applicability::MachineApplicable;
            let args = if let Some((receiver, args)) = args {
                // The first arg is the same kind as the receiver
                let explicit_args = if first_arg.is_some() {
                    std::iter::once(receiver).chain(args.iter()).collect::<Vec<_>>()
                } else {
                    // There is no `Self` kind to infer the arguments from
                    if has_unsuggestable_args {
                        applicability = Applicability::HasPlaceholders;
                    }
                    args.iter().collect()
                };
                format!(
                    "({}{})",
                    first_arg.unwrap_or(""),
                    explicit_args
                        .iter()
                        .map(|arg| self
                            .tcx
                            .sess
                            .source_map()
                            .span_to_snippet(arg.span)
                            .unwrap_or_else(|_| {
                                applicability = Applicability::HasPlaceholders;
                                "_".to_owned()
                            }))
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            } else {
                applicability = Applicability::HasPlaceholders;
                "(...)".to_owned()
            };
            err.span_suggestion(
                sugg_span,
                "use associated function syntax instead",
                format!("{}::{}{}", ty_str, item_name, args),
                applicability,
            );
        } else {
            err.help(&format!("try with `{}::{}`", ty_str, item_name,));
        }
    }

    /// Suggest calling a field with a type that implements the `Fn*` traits instead of a method with
    /// the same name as the field i.e. `(a.my_fn_ptr)(10)` instead of `a.my_fn_ptr(10)`.
    fn suggest_calling_field_as_fn(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        expr: &hir::Expr<'_>,
        item_name: Ident,
        err: &mut Diagnostic,
    ) -> bool {
        let tcx = self.tcx;
        let field_receiver = self.autoderef(span, rcvr_ty).find_map(|(ty, _)| match ty.kind() {
            ty::Adt(def, substs) if !def.is_enum() => {
                let variant = &def.non_enum_variant();
                tcx.find_field_index(item_name, variant).map(|index| {
                    let field = &variant.fields[index];
                    let field_ty = field.ty(tcx, substs);
                    (field, field_ty)
                })
            }
            _ => None,
        });
        if let Some((field, field_ty)) = field_receiver {
            let scope = tcx.parent_module_from_def_id(self.body_id);
            let is_accessible = field.vis.is_accessible_from(scope, tcx);

            if is_accessible {
                if self.is_fn_ty(field_ty, span) {
                    let expr_span = expr.span.to(item_name.span);
                    err.multipart_suggestion(
                        &format!(
                            "to call the function stored in `{}`, \
                                         surround the field access with parentheses",
                            item_name,
                        ),
                        vec![
                            (expr_span.shrink_to_lo(), '('.to_string()),
                            (expr_span.shrink_to_hi(), ')'.to_string()),
                        ],
                        Applicability::MachineApplicable,
                    );
                } else {
                    let call_expr = tcx.hir().expect_expr(tcx.hir().parent_id(expr.hir_id));

                    if let Some(span) = call_expr.span.trim_start(item_name.span) {
                        err.span_suggestion(
                            span,
                            "remove the arguments",
                            "",
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }

            let field_kind = if is_accessible { "field" } else { "private field" };
            err.span_label(item_name.span, format!("{}, not a method", field_kind));
            return true;
        }
        false
    }

    /// Suggest possible range with adding parentheses, for example:
    /// when encountering `0..1.map(|i| i + 1)` suggest `(0..1).map(|i| i + 1)`.
    fn suggest_wrapping_range_with_parens(
        &self,
        tcx: TyCtxt<'tcx>,
        actual: Ty<'tcx>,
        source: SelfSource<'tcx>,
        span: Span,
        item_name: Ident,
        ty_str: &str,
    ) -> bool {
        if let SelfSource::MethodCall(expr) = source {
            for (_, parent) in tcx.hir().parent_iter(expr.hir_id).take(5) {
                if let Node::Expr(parent_expr) = parent {
                    let lang_item = match parent_expr.kind {
                        ExprKind::Struct(ref qpath, _, _) => match **qpath {
                            QPath::LangItem(LangItem::Range, ..) => Some(LangItem::Range),
                            QPath::LangItem(LangItem::RangeTo, ..) => Some(LangItem::RangeTo),
                            QPath::LangItem(LangItem::RangeToInclusive, ..) => {
                                Some(LangItem::RangeToInclusive)
                            }
                            _ => None,
                        },
                        ExprKind::Call(ref func, _) => match func.kind {
                            // `..=` desugars into `::std::ops::RangeInclusive::new(...)`.
                            ExprKind::Path(QPath::LangItem(LangItem::RangeInclusiveNew, ..)) => {
                                Some(LangItem::RangeInclusiveStruct)
                            }
                            _ => None,
                        },
                        _ => None,
                    };

                    if lang_item.is_none() {
                        continue;
                    }

                    let span_included = match parent_expr.kind {
                        hir::ExprKind::Struct(_, eps, _) => {
                            eps.len() > 0 && eps.last().map_or(false, |ep| ep.span.contains(span))
                        }
                        // `..=` desugars into `::std::ops::RangeInclusive::new(...)`.
                        hir::ExprKind::Call(ref func, ..) => func.span.contains(span),
                        _ => false,
                    };

                    if !span_included {
                        continue;
                    }

                    let range_def_id = self.tcx.require_lang_item(lang_item.unwrap(), None);
                    let range_ty = self.tcx.type_of(range_def_id).subst(self.tcx, &[actual.into()]);

                    let pick = self.lookup_probe_for_diagnostic(
                        item_name,
                        range_ty,
                        expr,
                        ProbeScope::AllTraits,
                        None,
                    );
                    if pick.is_ok() {
                        let range_span = parent_expr.span.with_hi(expr.span.hi());
                        tcx.sess.emit_err(errors::MissingParentheseInRange {
                            span,
                            ty_str: ty_str.to_string(),
                            method_name: item_name.as_str().to_string(),
                            add_missing_parentheses: Some(errors::AddMissingParenthesesInRange {
                                func_name: item_name.name.as_str().to_string(),
                                left: range_span.shrink_to_lo(),
                                right: range_span.shrink_to_hi(),
                            }),
                        });
                        return true;
                    }
                }
            }
        }
        false
    }

    fn suggest_constraining_numerical_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        actual: Ty<'tcx>,
        source: SelfSource<'_>,
        span: Span,
        item_kind: &str,
        item_name: Ident,
        ty_str: &str,
    ) -> bool {
        let found_candidate = all_traits(self.tcx)
            .into_iter()
            .any(|info| self.associated_value(info.def_id, item_name).is_some());
        let found_assoc = |ty: Ty<'tcx>| {
            simplify_type(tcx, ty, TreatParams::AsCandidateKey)
                .and_then(|simp| {
                    tcx.incoherent_impls(simp)
                        .iter()
                        .find_map(|&id| self.associated_value(id, item_name))
                })
                .is_some()
        };
        let found_candidate = found_candidate
            || found_assoc(tcx.types.i8)
            || found_assoc(tcx.types.i16)
            || found_assoc(tcx.types.i32)
            || found_assoc(tcx.types.i64)
            || found_assoc(tcx.types.i128)
            || found_assoc(tcx.types.u8)
            || found_assoc(tcx.types.u16)
            || found_assoc(tcx.types.u32)
            || found_assoc(tcx.types.u64)
            || found_assoc(tcx.types.u128)
            || found_assoc(tcx.types.f32)
            || found_assoc(tcx.types.f32);
        if found_candidate
            && actual.is_numeric()
            && !actual.has_concrete_skeleton()
            && let SelfSource::MethodCall(expr) = source
        {
            let mut err = struct_span_err!(
                tcx.sess,
                span,
                E0689,
                "can't call {} `{}` on ambiguous numeric type `{}`",
                item_kind,
                item_name,
                ty_str
            );
            let concrete_type = if actual.is_integral() { "i32" } else { "f32" };
            match expr.kind {
                ExprKind::Lit(ref lit) => {
                    // numeric literal
                    let snippet = tcx
                        .sess
                        .source_map()
                        .span_to_snippet(lit.span)
                        .unwrap_or_else(|_| "<numeric literal>".to_owned());

                    // If this is a floating point literal that ends with '.',
                    // get rid of it to stop this from becoming a member access.
                    let snippet = snippet.strip_suffix('.').unwrap_or(&snippet);
                    err.span_suggestion(
                        lit.span,
                        &format!(
                            "you must specify a concrete type for this numeric value, \
                                         like `{}`",
                            concrete_type
                        ),
                        format!("{snippet}_{concrete_type}"),
                        Applicability::MaybeIncorrect,
                    );
                }
                ExprKind::Path(QPath::Resolved(_, path)) => {
                    // local binding
                    if let hir::def::Res::Local(hir_id) = path.res {
                        let span = tcx.hir().span(hir_id);
                        let filename = tcx.sess.source_map().span_to_filename(span);

                        let parent_node =
                            self.tcx.hir().get_parent(hir_id);
                        let msg = format!(
                            "you must specify a type for this binding, like `{}`",
                            concrete_type,
                        );

                        match (filename, parent_node) {
                            (
                                FileName::Real(_),
                                Node::Local(hir::Local {
                                    source: hir::LocalSource::Normal,
                                    ty,
                                    ..
                                }),
                            ) => {
                                let type_span = ty.map(|ty| ty.span.with_lo(span.hi())).unwrap_or(span.shrink_to_hi());
                                err.span_suggestion(
                                    // account for `let x: _ = 42;`
                                    //                   ^^^
                                    type_span,
                                    &msg,
                                    format!(": {concrete_type}"),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            _ => {
                                err.span_label(span, msg);
                            }
                        }
                    }
                }
                _ => {}
            }
            err.emit();
            return true;
        }
        false
    }

    /// For code `rect::area(...)`,
    /// if `rect` is a local variable and `area` is a valid assoc method for it,
    /// we try to suggest `rect.area()`
    pub(crate) fn suggest_assoc_method_call(&self, segs: &[PathSegment<'_>]) {
        debug!("suggest_assoc_method_call segs: {:?}", segs);
        let [seg1, seg2] = segs else { return; };
        let Some(mut diag) =
                self.tcx.sess.diagnostic().steal_diagnostic(seg1.ident.span, StashKey::CallAssocMethod)
                else { return };

        let map = self.infcx.tcx.hir();
        let body_id = self.tcx.hir().body_owned_by(self.body_id);
        let body = map.body(body_id);
        struct LetVisitor<'a> {
            result: Option<&'a hir::Expr<'a>>,
            ident_name: Symbol,
        }

        // FIXME: This really should be taking scoping, etc into account.
        impl<'v> Visitor<'v> for LetVisitor<'v> {
            fn visit_stmt(&mut self, ex: &'v hir::Stmt<'v>) {
                if let hir::StmtKind::Local(hir::Local { pat, init, .. }) = &ex.kind
                    && let Binding(_, _, ident, ..) = pat.kind
                    && ident.name == self.ident_name
                {
                    self.result = *init;
                } else {
                    hir::intravisit::walk_stmt(self, ex);
                }
            }
        }

        let mut visitor = LetVisitor { result: None, ident_name: seg1.ident.name };
        visitor.visit_body(&body);

        let parent = self.tcx.hir().parent_id(seg1.hir_id);
        if let Some(Node::Expr(call_expr)) = self.tcx.hir().find(parent)
            && let Some(expr) = visitor.result
            && let Some(self_ty) = self.node_ty_opt(expr.hir_id)
        {
            let probe = self.lookup_probe_for_diagnostic(
                seg2.ident,
                self_ty,
                call_expr,
                ProbeScope::TraitsInScope,
                None,
            );
            if probe.is_ok() {
                let sm = self.infcx.tcx.sess.source_map();
                diag.span_suggestion_verbose(
                    sm.span_extend_while(seg1.ident.span.shrink_to_hi(), |c| c == ':').unwrap(),
                    "you may have meant to call an instance method",
                    ".".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
        }
        diag.emit();
    }

    /// Suggest calling a method on a field i.e. `a.field.bar()` instead of `a.bar()`
    fn suggest_calling_method_on_field(
        &self,
        err: &mut Diagnostic,
        source: SelfSource<'tcx>,
        span: Span,
        actual: Ty<'tcx>,
        item_name: Ident,
        return_type: Option<Ty<'tcx>>,
    ) {
        if let SelfSource::MethodCall(expr) = source
        && let mod_id = self.tcx.parent_module(expr.hir_id).to_def_id()
        && let Some((fields, substs)) =
            self.get_field_candidates_considering_privacy(span, actual, mod_id)
        {
            let call_expr = self.tcx.hir().expect_expr(self.tcx.hir().parent_id(expr.hir_id));

            let lang_items = self.tcx.lang_items();
            let never_mention_traits = [
                lang_items.clone_trait(),
                lang_items.deref_trait(),
                lang_items.deref_mut_trait(),
                self.tcx.get_diagnostic_item(sym::AsRef),
                self.tcx.get_diagnostic_item(sym::AsMut),
                self.tcx.get_diagnostic_item(sym::Borrow),
                self.tcx.get_diagnostic_item(sym::BorrowMut),
            ];
            let candidate_fields: Vec<_> = fields
                .filter_map(|candidate_field| {
                    self.check_for_nested_field_satisfying(
                        span,
                        &|_, field_ty| {
                            self.lookup_probe_for_diagnostic(
                                item_name,
                                field_ty,
                                call_expr,
                                ProbeScope::TraitsInScope,
                                return_type,
                            )
                            .map_or(false, |pick| {
                                !never_mention_traits
                                    .iter()
                                    .flatten()
                                    .any(|def_id| self.tcx.parent(pick.item.def_id) == *def_id)
                            })
                        },
                        candidate_field,
                        substs,
                        vec![],
                        mod_id,
                    )
                })
                .map(|field_path| {
                    field_path
                        .iter()
                        .map(|id| id.name.to_ident_string())
                        .collect::<Vec<String>>()
                        .join(".")
                })
                .collect();

            let len = candidate_fields.len();
            if len > 0 {
                err.span_suggestions(
                    item_name.span.shrink_to_lo(),
                    format!(
                        "{} of the expressions' fields {} a method of the same name",
                        if len > 1 { "some" } else { "one" },
                        if len > 1 { "have" } else { "has" },
                    ),
                    candidate_fields.iter().map(|path| format!("{path}.")),
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }

    fn check_for_inner_self(
        &self,
        err: &mut Diagnostic,
        source: SelfSource<'tcx>,
        actual: Ty<'tcx>,
        item_name: Ident,
    ) {
        let tcx = self.tcx;
        let SelfSource::MethodCall(expr) = source else { return; };
        let call_expr = tcx.hir().expect_expr(tcx.hir().parent_id(expr.hir_id));

        let ty::Adt(kind, substs) = actual.kind() else { return; };
        match kind.adt_kind() {
            ty::AdtKind::Enum => {
                let matching_variants: Vec<_> = kind
                    .variants()
                    .iter()
                    .flat_map(|variant| {
                        let [field] = &variant.fields[..] else { return None; };
                        let field_ty = field.ty(tcx, substs);

                        // Skip `_`, since that'll just lead to ambiguity.
                        if self.resolve_vars_if_possible(field_ty).is_ty_var() {
                            return None;
                        }

                        self.lookup_probe_for_diagnostic(
                            item_name,
                            field_ty,
                            call_expr,
                            ProbeScope::TraitsInScope,
                            None,
                        )
                        .ok()
                        .map(|pick| (variant, field, pick))
                    })
                    .collect();

                let ret_ty_matches = |diagnostic_item| {
                    if let Some(ret_ty) = self
                        .ret_coercion
                        .as_ref()
                        .map(|c| self.resolve_vars_if_possible(c.borrow().expected_ty()))
                        && let ty::Adt(kind, _) = ret_ty.kind()
                        && tcx.get_diagnostic_item(diagnostic_item) == Some(kind.did())
                    {
                        true
                    } else {
                        false
                    }
                };

                match &matching_variants[..] {
                    [(_, field, pick)] => {
                        let self_ty = field.ty(tcx, substs);
                        err.span_note(
                            tcx.def_span(pick.item.def_id),
                            &format!("the method `{item_name}` exists on the type `{self_ty}`"),
                        );
                        let (article, kind, variant, question) =
                            if tcx.is_diagnostic_item(sym::Result, kind.did()) {
                                ("a", "Result", "Err", ret_ty_matches(sym::Result))
                            } else if tcx.is_diagnostic_item(sym::Option, kind.did()) {
                                ("an", "Option", "None", ret_ty_matches(sym::Option))
                            } else {
                                return;
                            };
                        if question {
                            err.span_suggestion_verbose(
                                expr.span.shrink_to_hi(),
                                format!(
                                    "use the `?` operator to extract the `{self_ty}` value, propagating \
                                    {article} `{kind}::{variant}` value to the caller"
                                ),
                                "?",
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_suggestion_verbose(
                                expr.span.shrink_to_hi(),
                                format!(
                                    "consider using `{kind}::expect` to unwrap the `{self_ty}` value, \
                                    panicking if the value is {article} `{kind}::{variant}`"
                                ),
                                ".expect(\"REASON\")",
                                Applicability::HasPlaceholders,
                            );
                        }
                    }
                    // FIXME(compiler-errors): Support suggestions for other matching enum variants
                    _ => {}
                }
            }
            // Target wrapper types - types that wrap or pretend to wrap another type,
            // perhaps this inner type is meant to be called?
            ty::AdtKind::Struct | ty::AdtKind::Union => {
                let [first] = ***substs else { return; };
                let ty::GenericArgKind::Type(ty) = first.unpack() else { return; };
                let Ok(pick) = self.lookup_probe_for_diagnostic(
                    item_name,
                    ty,
                    call_expr,
                    ProbeScope::TraitsInScope,
                    None,
                )  else { return; };

                let name = self.ty_to_value_string(actual);
                let inner_id = kind.did();
                let mutable = if let Some(AutorefOrPtrAdjustment::Autoref { mutbl, .. }) =
                    pick.autoref_or_ptr_adjustment
                {
                    Some(mutbl)
                } else {
                    None
                };

                if tcx.is_diagnostic_item(sym::LocalKey, inner_id) {
                    err.help("use `with` or `try_with` to access thread local storage");
                } else if Some(kind.did()) == tcx.lang_items().maybe_uninit() {
                    err.help(format!(
                        "if this `{name}` has been initialized, \
                        use one of the `assume_init` methods to access the inner value"
                    ));
                } else if tcx.is_diagnostic_item(sym::RefCell, inner_id) {
                    let (suggestion, borrow_kind, panic_if) = match mutable {
                        Some(Mutability::Not) => (".borrow()", "borrow", "a mutable borrow exists"),
                        Some(Mutability::Mut) => {
                            (".borrow_mut()", "mutably borrow", "any borrows exist")
                        }
                        None => return,
                    };
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "use `{suggestion}` to {borrow_kind} the `{ty}`, \
                            panicking if {panic_if}"
                        ),
                        suggestion,
                        Applicability::MaybeIncorrect,
                    );
                } else if tcx.is_diagnostic_item(sym::Mutex, inner_id) {
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "use `.lock().unwrap()` to borrow the `{ty}`, \
                            blocking the current thread until it can be acquired"
                        ),
                        ".lock().unwrap()",
                        Applicability::MaybeIncorrect,
                    );
                } else if tcx.is_diagnostic_item(sym::RwLock, inner_id) {
                    let (suggestion, borrow_kind) = match mutable {
                        Some(Mutability::Not) => (".read().unwrap()", "borrow"),
                        Some(Mutability::Mut) => (".write().unwrap()", "mutably borrow"),
                        None => return,
                    };
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "use `{suggestion}` to {borrow_kind} the `{ty}`, \
                            blocking the current thread until it can be acquired"
                        ),
                        suggestion,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    return;
                };

                err.span_note(
                    tcx.def_span(pick.item.def_id),
                    &format!("the method `{item_name}` exists on the type `{ty}`"),
                );
            }
        }
    }

    pub(crate) fn note_unmet_impls_on_type(
        &self,
        err: &mut Diagnostic,
        errors: Vec<FulfillmentError<'tcx>>,
    ) {
        let all_local_types_needing_impls =
            errors.iter().all(|e| match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::Clause::Trait(pred)) => match pred.self_ty().kind() {
                    ty::Adt(def, _) => def.did().is_local(),
                    _ => false,
                },
                _ => false,
            });
        let mut preds: Vec<_> = errors
            .iter()
            .filter_map(|e| match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::Clause::Trait(pred)) => Some(pred),
                _ => None,
            })
            .collect();
        preds.sort_by_key(|pred| (pred.def_id(), pred.self_ty()));
        let def_ids = preds
            .iter()
            .filter_map(|pred| match pred.self_ty().kind() {
                ty::Adt(def, _) => Some(def.did()),
                _ => None,
            })
            .collect::<FxHashSet<_>>();
        let mut spans: MultiSpan = def_ids
            .iter()
            .filter_map(|def_id| {
                let span = self.tcx.def_span(*def_id);
                if span.is_dummy() { None } else { Some(span) }
            })
            .collect::<Vec<_>>()
            .into();

        for pred in &preds {
            match pred.self_ty().kind() {
                ty::Adt(def, _) if def.did().is_local() => {
                    spans.push_span_label(
                        self.tcx.def_span(def.did()),
                        format!("must implement `{}`", pred.trait_ref.print_only_trait_path()),
                    );
                }
                _ => {}
            }
        }

        if all_local_types_needing_impls && spans.primary_span().is_some() {
            let msg = if preds.len() == 1 {
                format!(
                    "an implementation of `{}` might be missing for `{}`",
                    preds[0].trait_ref.print_only_trait_path(),
                    preds[0].self_ty()
                )
            } else {
                format!(
                    "the following type{} would have to `impl` {} required trait{} for this \
                     operation to be valid",
                    pluralize!(def_ids.len()),
                    if def_ids.len() == 1 { "its" } else { "their" },
                    pluralize!(preds.len()),
                )
            };
            err.span_note(spans, &msg);
        }

        let preds: Vec<_> = errors
            .iter()
            .map(|e| (e.obligation.predicate, None, Some(e.obligation.cause.clone())))
            .collect();
        self.suggest_derive(err, &preds);
    }

    pub fn suggest_derive(
        &self,
        err: &mut Diagnostic,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
    ) {
        let mut derives = Vec::<(String, Span, Symbol)>::new();
        let mut traits = Vec::new();
        for (pred, _, _) in unsatisfied_predicates {
            let ty::PredicateKind::Clause(ty::Clause::Trait(trait_pred)) = pred.kind().skip_binder() else { continue };
            let adt = match trait_pred.self_ty().ty_adt_def() {
                Some(adt) if adt.did().is_local() => adt,
                _ => continue,
            };
            if let Some(diagnostic_name) = self.tcx.get_diagnostic_name(trait_pred.def_id()) {
                let can_derive = match diagnostic_name {
                    sym::Default => !adt.is_enum(),
                    sym::Eq
                    | sym::PartialEq
                    | sym::Ord
                    | sym::PartialOrd
                    | sym::Clone
                    | sym::Copy
                    | sym::Hash
                    | sym::Debug => true,
                    _ => false,
                };
                if can_derive {
                    let self_name = trait_pred.self_ty().to_string();
                    let self_span = self.tcx.def_span(adt.did());
                    if let Some(poly_trait_ref) = pred.to_opt_poly_trait_pred() {
                        for super_trait in supertraits(self.tcx, poly_trait_ref.to_poly_trait_ref())
                        {
                            if let Some(parent_diagnostic_name) =
                                self.tcx.get_diagnostic_name(super_trait.def_id())
                            {
                                derives.push((
                                    self_name.clone(),
                                    self_span,
                                    parent_diagnostic_name,
                                ));
                            }
                        }
                    }
                    derives.push((self_name, self_span, diagnostic_name));
                } else {
                    traits.push(trait_pred.def_id());
                }
            } else {
                traits.push(trait_pred.def_id());
            }
        }
        traits.sort();
        traits.dedup();

        derives.sort();
        derives.dedup();

        let mut derives_grouped = Vec::<(String, Span, String)>::new();
        for (self_name, self_span, trait_name) in derives.into_iter() {
            if let Some((last_self_name, _, ref mut last_trait_names)) = derives_grouped.last_mut()
            {
                if last_self_name == &self_name {
                    last_trait_names.push_str(format!(", {}", trait_name).as_str());
                    continue;
                }
            }
            derives_grouped.push((self_name, self_span, trait_name.to_string()));
        }

        let len = traits.len();
        if len > 0 {
            let span =
                MultiSpan::from_spans(traits.iter().map(|&did| self.tcx.def_span(did)).collect());
            let mut names = format!("`{}`", self.tcx.def_path_str(traits[0]));
            for (i, &did) in traits.iter().enumerate().skip(1) {
                if len > 2 {
                    names.push_str(", ");
                }
                if i == len - 1 {
                    names.push_str(" and ");
                }
                names.push('`');
                names.push_str(&self.tcx.def_path_str(did));
                names.push('`');
            }
            err.span_note(
                span,
                &format!("the trait{} {} must be implemented", pluralize!(len), names),
            );
        }

        for (self_name, self_span, traits) in &derives_grouped {
            err.span_suggestion_verbose(
                self_span.shrink_to_lo(),
                &format!("consider annotating `{}` with `#[derive({})]`", self_name, traits),
                format!("#[derive({})]\n", traits),
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn check_for_deref_method(
        &self,
        err: &mut Diagnostic,
        self_source: SelfSource<'tcx>,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        expected: Expectation<'tcx>,
    ) {
        let SelfSource::QPath(ty) = self_source else { return; };
        for (deref_ty, _) in self.autoderef(rustc_span::DUMMY_SP, rcvr_ty).skip(1) {
            if let Ok(pick) = self.probe_for_name(
                Mode::Path,
                item_name,
                expected.only_has_type(self),
                IsSuggestion(true),
                deref_ty,
                ty.hir_id,
                ProbeScope::TraitsInScope,
            ) {
                if deref_ty.is_suggestable(self.tcx, true)
                    // If this method receives `&self`, then the provided
                    // argument _should_ coerce, so it's valid to suggest
                    // just changing the path.
                    && pick.item.fn_has_self_parameter
                    && let Some(self_ty) =
                        self.tcx.fn_sig(pick.item.def_id).subst_identity().inputs().skip_binder().get(0)
                    && self_ty.is_ref()
                {
                    let suggested_path = match deref_ty.kind() {
                        ty::Bool
                        | ty::Char
                        | ty::Int(_)
                        | ty::Uint(_)
                        | ty::Float(_)
                        | ty::Adt(_, _)
                        | ty::Str
                        | ty::Alias(ty::Projection, _)
                        | ty::Param(_) => format!("{deref_ty}"),
                        // we need to test something like  <&[_]>::len or <(&[u32])>::len
                        // and Vec::function();
                        // <&[_]>::len or <&[u32]>::len doesn't need an extra "<>" between
                        // but for Adt type like Vec::function()
                        // we would suggest <[_]>::function();
                        _ if self.tcx.sess.source_map().span_wrapped_by_angle_or_parentheses(ty.span)  => format!("{deref_ty}"),
                        _ => format!("<{deref_ty}>"),
                    };
                    err.span_suggestion_verbose(
                        ty.span,
                        format!("the function `{item_name}` is implemented on `{deref_ty}`"),
                        suggested_path,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    err.span_note(
                        ty.span,
                        format!("the function `{item_name}` is implemented on `{deref_ty}`"),
                    );
                }
                return;
            }
        }
    }

    /// Print out the type for use in value namespace.
    fn ty_to_value_string(&self, ty: Ty<'tcx>) -> String {
        match ty.kind() {
            ty::Adt(def, substs) => self.tcx.def_path_str_with_substs(def.did(), substs),
            _ => self.ty_to_string(ty),
        }
    }

    fn suggest_await_before_method(
        &self,
        err: &mut Diagnostic,
        item_name: Ident,
        ty: Ty<'tcx>,
        call: &hir::Expr<'_>,
        span: Span,
        return_type: Option<Ty<'tcx>>,
    ) {
        let output_ty = match self.get_impl_future_output_ty(ty) {
            Some(output_ty) => self.resolve_vars_if_possible(output_ty),
            _ => return,
        };
        let method_exists =
            self.method_exists(item_name, output_ty, call.hir_id, true, return_type);
        debug!("suggest_await_before_method: is_method_exist={}", method_exists);
        if method_exists {
            err.span_suggestion_verbose(
                span.shrink_to_lo(),
                "consider `await`ing on the `Future` and calling the method on its `Output`",
                "await.",
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn suggest_use_candidates(&self, err: &mut Diagnostic, msg: String, candidates: Vec<DefId>) {
        let parent_map = self.tcx.visible_parent_map(());

        // Separate out candidates that must be imported with a glob, because they are named `_`
        // and cannot be referred with their identifier.
        let (candidates, globs): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|trait_did| {
            if let Some(parent_did) = parent_map.get(trait_did) {
                // If the item is re-exported as `_`, we should suggest a glob-import instead.
                if *parent_did != self.tcx.parent(*trait_did)
                    && self
                        .tcx
                        .module_children(*parent_did)
                        .iter()
                        .filter(|child| child.res.opt_def_id() == Some(*trait_did))
                        .all(|child| child.ident.name == kw::Underscore)
                {
                    return false;
                }
            }

            true
        });

        let module_did = self.tcx.parent_module_from_def_id(self.body_id);
        let (module, _, _) = self.tcx.hir().get_module(module_did);
        let span = module.spans.inject_use_span;

        let path_strings = candidates.iter().map(|trait_did| {
            format!("use {};\n", with_crate_prefix!(self.tcx.def_path_str(*trait_did)),)
        });

        let glob_path_strings = globs.iter().map(|trait_did| {
            let parent_did = parent_map.get(trait_did).unwrap();
            format!(
                "use {}::*; // trait {}\n",
                with_crate_prefix!(self.tcx.def_path_str(*parent_did)),
                self.tcx.item_name(*trait_did),
            )
        });

        err.span_suggestions(
            span,
            &msg,
            path_strings.chain(glob_path_strings),
            Applicability::MaybeIncorrect,
        );
    }

    fn suggest_valid_traits(
        &self,
        err: &mut Diagnostic,
        valid_out_of_scope_traits: Vec<DefId>,
    ) -> bool {
        if !valid_out_of_scope_traits.is_empty() {
            let mut candidates = valid_out_of_scope_traits;
            candidates.sort();
            candidates.dedup();

            // `TryFrom` and `FromIterator` have no methods
            let edition_fix = candidates
                .iter()
                .find(|did| self.tcx.is_diagnostic_item(sym::TryInto, **did))
                .copied();

            err.help("items from traits can only be used if the trait is in scope");
            let msg = format!(
                "the following {traits_are} implemented but not in scope; \
                 perhaps add a `use` for {one_of_them}:",
                traits_are = if candidates.len() == 1 { "trait is" } else { "traits are" },
                one_of_them = if candidates.len() == 1 { "it" } else { "one of them" },
            );

            self.suggest_use_candidates(err, msg, candidates);
            if let Some(did) = edition_fix {
                err.note(&format!(
                    "'{}' is included in the prelude starting in Edition 2021",
                    with_crate_prefix!(self.tcx.def_path_str(did))
                ));
            }

            true
        } else {
            false
        }
    }

    fn suggest_traits_to_import(
        &self,
        err: &mut Diagnostic,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        inputs_len: Option<usize>,
        source: SelfSource<'tcx>,
        valid_out_of_scope_traits: Vec<DefId>,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
        static_candidates: &[CandidateSource],
        unsatisfied_bounds: bool,
        return_type: Option<Ty<'tcx>>,
    ) {
        let mut alt_rcvr_sugg = false;
        if let (SelfSource::MethodCall(rcvr), false) = (source, unsatisfied_bounds) {
            debug!(
                "suggest_traits_to_import: span={:?}, item_name={:?}, rcvr_ty={:?}, rcvr={:?}",
                span, item_name, rcvr_ty, rcvr
            );
            let skippable = [
                self.tcx.lang_items().clone_trait(),
                self.tcx.lang_items().deref_trait(),
                self.tcx.lang_items().deref_mut_trait(),
                self.tcx.lang_items().drop_trait(),
                self.tcx.get_diagnostic_item(sym::AsRef),
            ];
            // Try alternative arbitrary self types that could fulfill this call.
            // FIXME: probe for all types that *could* be arbitrary self-types, not
            // just this list.
            for (rcvr_ty, post) in &[
                (rcvr_ty, ""),
                (self.tcx.mk_mut_ref(self.tcx.lifetimes.re_erased, rcvr_ty), "&mut "),
                (self.tcx.mk_imm_ref(self.tcx.lifetimes.re_erased, rcvr_ty), "&"),
            ] {
                match self.lookup_probe_for_diagnostic(
                    item_name,
                    *rcvr_ty,
                    rcvr,
                    ProbeScope::AllTraits,
                    return_type,
                ) {
                    Ok(pick) => {
                        // If the method is defined for the receiver we have, it likely wasn't `use`d.
                        // We point at the method, but we just skip the rest of the check for arbitrary
                        // self types and rely on the suggestion to `use` the trait from
                        // `suggest_valid_traits`.
                        let did = Some(pick.item.container_id(self.tcx));
                        let skip = skippable.contains(&did);
                        if pick.autoderefs == 0 && !skip {
                            err.span_label(
                                pick.item.ident(self.tcx).span,
                                &format!("the method is available for `{}` here", rcvr_ty),
                            );
                        }
                        break;
                    }
                    Err(MethodError::Ambiguity(_)) => {
                        // If the method is defined (but ambiguous) for the receiver we have, it is also
                        // likely we haven't `use`d it. It may be possible that if we `Box`/`Pin`/etc.
                        // the receiver, then it might disambiguate this method, but I think these
                        // suggestions are generally misleading (see #94218).
                        break;
                    }
                    Err(_) => (),
                }

                for (rcvr_ty, pre) in &[
                    (self.tcx.mk_lang_item(*rcvr_ty, LangItem::OwnedBox), "Box::new"),
                    (self.tcx.mk_lang_item(*rcvr_ty, LangItem::Pin), "Pin::new"),
                    (self.tcx.mk_diagnostic_item(*rcvr_ty, sym::Arc), "Arc::new"),
                    (self.tcx.mk_diagnostic_item(*rcvr_ty, sym::Rc), "Rc::new"),
                ] {
                    if let Some(new_rcvr_t) = *rcvr_ty
                        && let Ok(pick) = self.lookup_probe_for_diagnostic(
                            item_name,
                            new_rcvr_t,
                            rcvr,
                            ProbeScope::AllTraits,
                            return_type,
                        )
                    {
                        debug!("try_alt_rcvr: pick candidate {:?}", pick);
                        let did = Some(pick.item.container_id(self.tcx));
                        // We don't want to suggest a container type when the missing
                        // method is `.clone()` or `.deref()` otherwise we'd suggest
                        // `Arc::new(foo).clone()`, which is far from what the user wants.
                        // Explicitly ignore the `Pin::as_ref()` method as `Pin` does not
                        // implement the `AsRef` trait.
                        let skip = skippable.contains(&did)
                            || (("Pin::new" == *pre) && (sym::as_ref == item_name.name))
                            || inputs_len.map_or(false, |inputs_len| pick.item.kind == ty::AssocKind::Fn && self.tcx.fn_sig(pick.item.def_id).skip_binder().skip_binder().inputs().len() != inputs_len);
                        // Make sure the method is defined for the *actual* receiver: we don't
                        // want to treat `Box<Self>` as a receiver if it only works because of
                        // an autoderef to `&self`
                        if pick.autoderefs == 0 && !skip {
                            err.span_label(
                                pick.item.ident(self.tcx).span,
                                &format!("the method is available for `{}` here", new_rcvr_t),
                            );
                            err.multipart_suggestion(
                                "consider wrapping the receiver expression with the \
                                    appropriate type",
                                vec![
                                    (rcvr.span.shrink_to_lo(), format!("{}({}", pre, post)),
                                    (rcvr.span.shrink_to_hi(), ")".to_string()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                            // We don't care about the other suggestions.
                            alt_rcvr_sugg = true;
                        }
                    }
                }
            }
        }
        if self.suggest_valid_traits(err, valid_out_of_scope_traits) {
            return;
        }

        let type_is_local = self.type_derefs_to_local(span, rcvr_ty, source);

        let mut arbitrary_rcvr = vec![];
        // There are no traits implemented, so lets suggest some traits to
        // implement, by finding ones that have the item name, and are
        // legal to implement.
        let mut candidates = all_traits(self.tcx)
            .into_iter()
            // Don't issue suggestions for unstable traits since they're
            // unlikely to be implementable anyway
            .filter(|info| match self.tcx.lookup_stability(info.def_id) {
                Some(attr) => attr.level.is_stable(),
                None => true,
            })
            .filter(|info| {
                // Static candidates are already implemented, and known not to work
                // Do not suggest them again
                static_candidates.iter().all(|sc| match *sc {
                    CandidateSource::Trait(def_id) => def_id != info.def_id,
                    CandidateSource::Impl(def_id) => {
                        self.tcx.trait_id_of_impl(def_id) != Some(info.def_id)
                    }
                })
            })
            .filter(|info| {
                // We approximate the coherence rules to only suggest
                // traits that are legal to implement by requiring that
                // either the type or trait is local. Multi-dispatch means
                // this isn't perfect (that is, there are cases when
                // implementing a trait would be legal but is rejected
                // here).
                unsatisfied_predicates.iter().all(|(p, _, _)| {
                    match p.kind().skip_binder() {
                        // Hide traits if they are present in predicates as they can be fixed without
                        // having to implement them.
                        ty::PredicateKind::Clause(ty::Clause::Trait(t)) => {
                            t.def_id() == info.def_id
                        }
                        ty::PredicateKind::Clause(ty::Clause::Projection(p)) => {
                            p.projection_ty.def_id == info.def_id
                        }
                        _ => false,
                    }
                }) && (type_is_local || info.def_id.is_local())
                    && !self.tcx.trait_is_auto(info.def_id)
                    && self
                        .associated_value(info.def_id, item_name)
                        .filter(|item| {
                            if let ty::AssocKind::Fn = item.kind {
                                let id = item
                                    .def_id
                                    .as_local()
                                    .map(|def_id| self.tcx.hir().local_def_id_to_hir_id(def_id));
                                if let Some(hir::Node::TraitItem(hir::TraitItem {
                                    kind: hir::TraitItemKind::Fn(fn_sig, method),
                                    ..
                                })) = id.map(|id| self.tcx.hir().get(id))
                                {
                                    let self_first_arg = match method {
                                        hir::TraitFn::Required([ident, ..]) => {
                                            ident.name == kw::SelfLower
                                        }
                                        hir::TraitFn::Provided(body_id) => {
                                            self.tcx.hir().body(*body_id).params.first().map_or(
                                                false,
                                                |param| {
                                                    matches!(
                                                        param.pat.kind,
                                                        hir::PatKind::Binding(_, _, ident, _)
                                                            if ident.name == kw::SelfLower
                                                    )
                                                },
                                            )
                                        }
                                        _ => false,
                                    };

                                    if !fn_sig.decl.implicit_self.has_implicit_self()
                                        && self_first_arg
                                    {
                                        if let Some(ty) = fn_sig.decl.inputs.get(0) {
                                            arbitrary_rcvr.push(ty.span);
                                        }
                                        return false;
                                    }
                                }
                            }
                            // We only want to suggest public or local traits (#45781).
                            item.visibility(self.tcx).is_public() || info.def_id.is_local()
                        })
                        .is_some()
            })
            .collect::<Vec<_>>();
        for span in &arbitrary_rcvr {
            err.span_label(
                *span,
                "the method might not be found because of this arbitrary self type",
            );
        }
        if alt_rcvr_sugg {
            return;
        }

        if !candidates.is_empty() {
            // Sort from most relevant to least relevant.
            candidates.sort_by_key(|&info| cmp::Reverse(info));
            candidates.dedup();

            let param_type = match rcvr_ty.kind() {
                ty::Param(param) => Some(param),
                ty::Ref(_, ty, _) => match ty.kind() {
                    ty::Param(param) => Some(param),
                    _ => None,
                },
                _ => None,
            };
            err.help(if param_type.is_some() {
                "items from traits can only be used if the type parameter is bounded by the trait"
            } else {
                "items from traits can only be used if the trait is implemented and in scope"
            });
            let candidates_len = candidates.len();
            let message = |action| {
                format!(
                    "the following {traits_define} an item `{name}`, perhaps you need to {action} \
                     {one_of_them}:",
                    traits_define =
                        if candidates_len == 1 { "trait defines" } else { "traits define" },
                    action = action,
                    one_of_them = if candidates_len == 1 { "it" } else { "one of them" },
                    name = item_name,
                )
            };
            // Obtain the span for `param` and use it for a structured suggestion.
            if let Some(param) = param_type {
                let generics = self.tcx.generics_of(self.body_id.to_def_id());
                let type_param = generics.type_param(param, self.tcx);
                let hir = self.tcx.hir();
                if let Some(def_id) = type_param.def_id.as_local() {
                    let id = hir.local_def_id_to_hir_id(def_id);
                    // Get the `hir::Param` to verify whether it already has any bounds.
                    // We do this to avoid suggesting code that ends up as `T: FooBar`,
                    // instead we suggest `T: Foo + Bar` in that case.
                    match hir.get(id) {
                        Node::GenericParam(param) => {
                            enum Introducer {
                                Plus,
                                Colon,
                                Nothing,
                            }
                            let ast_generics = hir.get_generics(id.owner.def_id).unwrap();
                            let (sp, mut introducer) = if let Some(span) =
                                ast_generics.bounds_span_for_suggestions(def_id)
                            {
                                (span, Introducer::Plus)
                            } else if let Some(colon_span) = param.colon_span {
                                (colon_span.shrink_to_hi(), Introducer::Nothing)
                            } else {
                                (param.span.shrink_to_hi(), Introducer::Colon)
                            };
                            if matches!(
                                param.kind,
                                hir::GenericParamKind::Type { synthetic: true, .. },
                            ) {
                                introducer = Introducer::Plus
                            }
                            let trait_def_ids: FxHashSet<DefId> = ast_generics
                                .bounds_for_param(def_id)
                                .flat_map(|bp| bp.bounds.iter())
                                .filter_map(|bound| bound.trait_ref()?.trait_def_id())
                                .collect();
                            if !candidates.iter().any(|t| trait_def_ids.contains(&t.def_id)) {
                                err.span_suggestions(
                                    sp,
                                    &message(format!(
                                        "restrict type parameter `{}` with",
                                        param.name.ident(),
                                    )),
                                    candidates.iter().map(|t| {
                                        format!(
                                            "{} {}",
                                            match introducer {
                                                Introducer::Plus => " +",
                                                Introducer::Colon => ":",
                                                Introducer::Nothing => "",
                                            },
                                            self.tcx.def_path_str(t.def_id),
                                        )
                                    }),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            return;
                        }
                        Node::Item(hir::Item {
                            kind: hir::ItemKind::Trait(.., bounds, _),
                            ident,
                            ..
                        }) => {
                            let (sp, sep, article) = if bounds.is_empty() {
                                (ident.span.shrink_to_hi(), ":", "a")
                            } else {
                                (bounds.last().unwrap().span().shrink_to_hi(), " +", "another")
                            };
                            err.span_suggestions(
                                sp,
                                &message(format!("add {} supertrait for", article)),
                                candidates.iter().map(|t| {
                                    format!("{} {}", sep, self.tcx.def_path_str(t.def_id),)
                                }),
                                Applicability::MaybeIncorrect,
                            );
                            return;
                        }
                        _ => {}
                    }
                }
            }

            let (potential_candidates, explicitly_negative) = if param_type.is_some() {
                // FIXME: Even though negative bounds are not implemented, we could maybe handle
                // cases where a positive bound implies a negative impl.
                (candidates, Vec::new())
            } else if let Some(simp_rcvr_ty) =
                simplify_type(self.tcx, rcvr_ty, TreatParams::ForLookup)
            {
                let mut potential_candidates = Vec::new();
                let mut explicitly_negative = Vec::new();
                for candidate in candidates {
                    // Check if there's a negative impl of `candidate` for `rcvr_ty`
                    if self
                        .tcx
                        .all_impls(candidate.def_id)
                        .filter(|imp_did| {
                            self.tcx.impl_polarity(*imp_did) == ty::ImplPolarity::Negative
                        })
                        .any(|imp_did| {
                            let imp = self.tcx.impl_trait_ref(imp_did).unwrap().subst_identity();
                            let imp_simp =
                                simplify_type(self.tcx, imp.self_ty(), TreatParams::ForLookup);
                            imp_simp.map_or(false, |s| s == simp_rcvr_ty)
                        })
                    {
                        explicitly_negative.push(candidate);
                    } else {
                        potential_candidates.push(candidate);
                    }
                }
                (potential_candidates, explicitly_negative)
            } else {
                // We don't know enough about `recv_ty` to make proper suggestions.
                (candidates, Vec::new())
            };

            let action = if let Some(param) = param_type {
                format!("restrict type parameter `{}` with", param)
            } else {
                // FIXME: it might only need to be imported into scope, not implemented.
                "implement".to_string()
            };
            match &potential_candidates[..] {
                [] => {}
                [trait_info] if trait_info.def_id.is_local() => {
                    err.span_note(
                        self.tcx.def_span(trait_info.def_id),
                        &format!(
                            "`{}` defines an item `{}`, perhaps you need to {} it",
                            self.tcx.def_path_str(trait_info.def_id),
                            item_name,
                            action
                        ),
                    );
                }
                trait_infos => {
                    let mut msg = message(action);
                    for (i, trait_info) in trait_infos.iter().enumerate() {
                        msg.push_str(&format!(
                            "\ncandidate #{}: `{}`",
                            i + 1,
                            self.tcx.def_path_str(trait_info.def_id),
                        ));
                    }
                    err.note(&msg);
                }
            }
            match &explicitly_negative[..] {
                [] => {}
                [trait_info] => {
                    let msg = format!(
                        "the trait `{}` defines an item `{}`, but is explicitly unimplemented",
                        self.tcx.def_path_str(trait_info.def_id),
                        item_name
                    );
                    err.note(&msg);
                }
                trait_infos => {
                    let mut msg = format!(
                        "the following traits define an item `{}`, but are explicitly unimplemented:",
                        item_name
                    );
                    for trait_info in trait_infos {
                        msg.push_str(&format!("\n{}", self.tcx.def_path_str(trait_info.def_id)));
                    }
                    err.note(&msg);
                }
            }
        }
    }

    /// issue #102320, for `unwrap_or` with closure as argument, suggest `unwrap_or_else`
    /// FIXME: currently not working for suggesting `map_or_else`, see #102408
    pub(crate) fn suggest_else_fn_with_closure(
        &self,
        err: &mut Diagnostic,
        expr: &hir::Expr<'_>,
        found: Ty<'tcx>,
        expected: Ty<'tcx>,
    ) -> bool {
        let Some((_def_id_or_name, output, _inputs)) =
            self.extract_callable_info(found) else {
                return false;
        };

        if !self.can_coerce(output, expected) {
            return false;
        }

        let parent = self.tcx.hir().parent_id(expr.hir_id);
        if  let Some(Node::Expr(call_expr)) = self.tcx.hir().find(parent) &&
            let hir::ExprKind::MethodCall(
                hir::PathSegment { ident: method_name, .. },
                self_expr,
                args,
                ..,
             ) = call_expr.kind &&
            let Some(self_ty) = self.typeck_results.borrow().expr_ty_opt(self_expr) {
            let new_name = Ident {
                name: Symbol::intern(&format!("{}_else", method_name.as_str())),
                span: method_name.span,
            };
            let probe = self.lookup_probe_for_diagnostic(
                new_name,
                self_ty,
                self_expr,
                ProbeScope::TraitsInScope,
                Some(expected),
            );

            // check the method arguments number
            if let Ok(pick) = probe &&
                let fn_sig = self.tcx.fn_sig(pick.item.def_id) &&
                let fn_args = fn_sig.skip_binder().skip_binder().inputs() &&
                fn_args.len() == args.len() + 1 {
                err.span_suggestion_verbose(
                    method_name.span.shrink_to_hi(),
                    &format!("try calling `{}` instead", new_name.name.as_str()),
                    "_else",
                    Applicability::MaybeIncorrect,
                );
                return true;
            }
        }
        false
    }

    /// Checks whether there is a local type somewhere in the chain of
    /// autoderefs of `rcvr_ty`.
    fn type_derefs_to_local(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        source: SelfSource<'tcx>,
    ) -> bool {
        fn is_local(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Adt(def, _) => def.did().is_local(),
                ty::Foreign(did) => did.is_local(),
                ty::Dynamic(tr, ..) => tr.principal().map_or(false, |d| d.def_id().is_local()),
                ty::Param(_) => true,

                // Everything else (primitive types, etc.) is effectively
                // non-local (there are "edge" cases, e.g., `(LocalType,)`, but
                // the noise from these sort of types is usually just really
                // annoying, rather than any sort of help).
                _ => false,
            }
        }

        // This occurs for UFCS desugaring of `T::method`, where there is no
        // receiver expression for the method call, and thus no autoderef.
        if let SelfSource::QPath(_) = source {
            return is_local(self.resolve_vars_with_obligations(rcvr_ty));
        }

        self.autoderef(span, rcvr_ty).any(|(ty, _)| is_local(ty))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SelfSource<'a> {
    QPath(&'a hir::Ty<'a>),
    MethodCall(&'a hir::Expr<'a> /* rcvr */),
}

#[derive(Copy, Clone)]
pub struct TraitInfo {
    pub def_id: DefId,
}

impl PartialEq for TraitInfo {
    fn eq(&self, other: &TraitInfo) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for TraitInfo {}
impl PartialOrd for TraitInfo {
    fn partial_cmp(&self, other: &TraitInfo) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TraitInfo {
    fn cmp(&self, other: &TraitInfo) -> Ordering {
        // Local crates are more important than remote ones (local:
        // `cnum == 0`), and otherwise we throw in the defid for totality.

        let lhs = (other.def_id.krate, other.def_id);
        let rhs = (self.def_id.krate, self.def_id);
        lhs.cmp(&rhs)
    }
}

/// Retrieves all traits in this crate and any dependent crates,
/// and wraps them into `TraitInfo` for custom sorting.
pub fn all_traits(tcx: TyCtxt<'_>) -> Vec<TraitInfo> {
    tcx.all_traits().map(|def_id| TraitInfo { def_id }).collect()
}

fn print_disambiguation_help<'tcx>(
    item_name: Ident,
    args: Option<(&'tcx hir::Expr<'tcx>, &'tcx [hir::Expr<'tcx>])>,
    err: &mut Diagnostic,
    trait_name: String,
    rcvr_ty: Ty<'_>,
    kind: ty::AssocKind,
    def_kind_descr: &'static str,
    span: Span,
    candidate: Option<usize>,
    source_map: &source_map::SourceMap,
    fn_has_self_parameter: bool,
) {
    let mut applicability = Applicability::MachineApplicable;
    let (span, sugg) = if let (ty::AssocKind::Fn, Some((receiver, args))) = (kind, args) {
        let args = format!(
            "({}{})",
            rcvr_ty.ref_mutability().map_or("", |mutbl| mutbl.ref_prefix_str()),
            std::iter::once(receiver)
                .chain(args.iter())
                .map(|arg| source_map.span_to_snippet(arg.span).unwrap_or_else(|_| {
                    applicability = Applicability::HasPlaceholders;
                    "_".to_owned()
                }))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let trait_name = if !fn_has_self_parameter {
            format!("<{} as {}>", rcvr_ty, trait_name)
        } else {
            trait_name
        };
        (span, format!("{}::{}{}", trait_name, item_name, args))
    } else {
        (span.with_hi(item_name.span.lo()), format!("<{} as {}>::", rcvr_ty, trait_name))
    };
    err.span_suggestion_verbose(
        span,
        &format!(
            "disambiguate the {} for {}",
            def_kind_descr,
            if let Some(candidate) = candidate {
                format!("candidate #{}", candidate)
            } else {
                "the candidate".to_string()
            },
        ),
        sugg,
        applicability,
    );
}

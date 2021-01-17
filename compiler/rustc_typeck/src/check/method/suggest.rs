//! Give useful errors and suggestions to users when an item can't be
//! found or is otherwise invalid.

use crate::check::FnCtxt;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Namespace, Res};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::intravisit;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ExprKind, Node, QPath};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::hir::map as hir_map;
use rustc_middle::ty::fast_reject::simplify_type;
use rustc_middle::ty::print::with_crate_prefix;
use rustc_middle::ty::{
    self, ToPolyTraitRef, ToPredicate, Ty, TyCtxt, TypeFoldable, WithConstness,
};
use rustc_span::lev_distance;
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::{source_map, FileName, Span};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::Obligation;

use std::cmp::Ordering;

use super::probe::Mode;
use super::{CandidateSource, MethodError, NoMatchData};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn is_fn_ty(&self, ty: Ty<'tcx>, span: Span) -> bool {
        let tcx = self.tcx;
        match ty.kind() {
            // Not all of these (e.g., unsafe fns) implement `FnOnce`,
            // so we look for these beforehand.
            ty::Closure(..) | ty::FnDef(..) | ty::FnPtr(_) => true,
            // If it's not a simple function, look for things which implement `FnOnce`.
            _ => {
                let fn_once = match tcx.lang_items().require(LangItem::FnOnce) {
                    Ok(fn_once) => fn_once,
                    Err(..) => return false,
                };

                self.autoderef(span, ty).any(|(ty, _)| {
                    self.probe(|_| {
                        let fn_once_substs = tcx.mk_substs_trait(
                            ty,
                            &[self
                                .next_ty_var(TypeVariableOrigin {
                                    kind: TypeVariableOriginKind::MiscVariable,
                                    span,
                                })
                                .into()],
                        );
                        let trait_ref = ty::TraitRef::new(fn_once, fn_once_substs);
                        let poly_trait_ref = trait_ref.to_poly_trait_ref();
                        let obligation = Obligation::misc(
                            span,
                            self.body_id,
                            self.param_env,
                            poly_trait_ref.without_const().to_predicate(tcx),
                        );
                        self.predicate_may_hold(&obligation)
                    })
                })
            }
        }
    }

    pub fn report_method_error<'b>(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        source: SelfSource<'b>,
        error: MethodError<'tcx>,
        args: Option<&'tcx [hir::Expr<'tcx>]>,
    ) -> Option<DiagnosticBuilder<'_>> {
        let orig_span = span;
        let mut span = span;
        // Avoid suggestions when we don't know what's going on.
        if rcvr_ty.references_error() {
            return None;
        }

        let report_candidates = |span: Span,
                                 err: &mut DiagnosticBuilder<'_>,
                                 mut sources: Vec<CandidateSource>,
                                 sugg_span: Span| {
            sources.sort();
            sources.dedup();
            // Dynamic limit to avoid hiding just one candidate, which is silly.
            let limit = if sources.len() == 5 { 5 } else { 4 };

            for (idx, source) in sources.iter().take(limit).enumerate() {
                match *source {
                    CandidateSource::ImplSource(impl_did) => {
                        // Provide the best span we can. Use the item, if local to crate, else
                        // the impl, if local to crate (item may be defaulted), else nothing.
                        let item = match self
                            .associated_item(impl_did, item_name, Namespace::ValueNS)
                            .or_else(|| {
                                let impl_trait_ref = self.tcx.impl_trait_ref(impl_did)?;
                                self.associated_item(
                                    impl_trait_ref.def_id,
                                    item_name,
                                    Namespace::ValueNS,
                                )
                            }) {
                            Some(item) => item,
                            None => continue,
                        };
                        let note_span = self
                            .tcx
                            .hir()
                            .span_if_local(item.def_id)
                            .or_else(|| self.tcx.hir().span_if_local(impl_did));

                        let impl_ty = self.tcx.at(span).type_of(impl_did);

                        let insertion = match self.tcx.impl_trait_ref(impl_did) {
                            None => String::new(),
                            Some(trait_ref) => format!(
                                " of the trait `{}`",
                                self.tcx.def_path_str(trait_ref.def_id)
                            ),
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
                            err.span_note(
                                self.tcx.sess.source_map().guess_head_span(note_span),
                                &note_str,
                            );
                        } else {
                            err.note(&note_str);
                        }
                        if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_did) {
                            let path = self.tcx.def_path_str(trait_ref.def_id);

                            let ty = match item.kind {
                                ty::AssocKind::Const | ty::AssocKind::Type => rcvr_ty,
                                ty::AssocKind::Fn => self
                                    .tcx
                                    .fn_sig(item.def_id)
                                    .inputs()
                                    .skip_binder()
                                    .get(0)
                                    .filter(|ty| ty.is_region_ptr() && !rcvr_ty.is_region_ptr())
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
                                item.def_id,
                                sugg_span,
                                idx,
                                self.tcx.sess.source_map(),
                            );
                        }
                    }
                    CandidateSource::TraitSource(trait_did) => {
                        let item =
                            match self.associated_item(trait_did, item_name, Namespace::ValueNS) {
                                Some(item) => item,
                                None => continue,
                            };
                        let item_span = self
                            .tcx
                            .sess
                            .source_map()
                            .guess_head_span(self.tcx.def_span(item.def_id));
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
                        let path = self.tcx.def_path_str(trait_did);
                        print_disambiguation_help(
                            item_name,
                            args,
                            err,
                            path,
                            rcvr_ty,
                            item.kind,
                            item.def_id,
                            sugg_span,
                            idx,
                            self.tcx.sess.source_map(),
                        );
                    }
                }
            }
            if sources.len() > limit {
                err.note(&format!("and {} others", sources.len() - limit));
            }
        };

        let sugg_span = if let SelfSource::MethodCall(expr) = source {
            // Given `foo.bar(baz)`, `expr` is `bar`, but we want to point to the whole thing.
            self.tcx.hir().expect_expr(self.tcx.hir().get_parent_node(expr.hir_id)).span
        } else {
            span
        };

        match error {
            MethodError::NoMatch(NoMatchData {
                static_candidates: static_sources,
                unsatisfied_predicates,
                out_of_scope_traits,
                lev_candidate,
                mode,
            }) => {
                let tcx = self.tcx;

                let actual = self.resolve_vars_if_possible(rcvr_ty);
                let ty_str = self.ty_to_string(actual);
                let is_method = mode == Mode::MethodCall;
                let item_kind = if is_method {
                    "method"
                } else if actual.is_enum() {
                    "variant or associated item"
                } else {
                    match (item_name.as_str().chars().next(), actual.is_fresh_ty()) {
                        (Some(name), false) if name.is_lowercase() => "function or associated item",
                        (Some(_), false) => "associated item",
                        (Some(_), true) | (None, false) => "variant or associated item",
                        (None, true) => "variant",
                    }
                };
                let mut err = if !actual.references_error() {
                    // Suggest clamping down the type if the method that is being attempted to
                    // be used exists at all, and the type is an ambiguous numeric type
                    // ({integer}/{float}).
                    let mut candidates = all_traits(self.tcx).into_iter().filter_map(|info| {
                        self.associated_item(info.def_id, item_name, Namespace::ValueNS)
                    });
                    // There are methods that are defined on the primitive types and won't be
                    // found when exploring `all_traits`, but we also need them to be acurate on
                    // our suggestions (#47759).
                    let fund_assoc = |opt_def_id: Option<DefId>| {
                        opt_def_id
                            .and_then(|id| self.associated_item(id, item_name, Namespace::ValueNS))
                            .is_some()
                    };
                    let lang_items = tcx.lang_items();
                    let found_candidate = candidates.next().is_some()
                        || fund_assoc(lang_items.i8_impl())
                        || fund_assoc(lang_items.i16_impl())
                        || fund_assoc(lang_items.i32_impl())
                        || fund_assoc(lang_items.i64_impl())
                        || fund_assoc(lang_items.i128_impl())
                        || fund_assoc(lang_items.u8_impl())
                        || fund_assoc(lang_items.u16_impl())
                        || fund_assoc(lang_items.u32_impl())
                        || fund_assoc(lang_items.u64_impl())
                        || fund_assoc(lang_items.u128_impl())
                        || fund_assoc(lang_items.f32_impl())
                        || fund_assoc(lang_items.f32_runtime_impl())
                        || fund_assoc(lang_items.f64_impl())
                        || fund_assoc(lang_items.f64_runtime_impl());
                    if let (true, false, SelfSource::MethodCall(expr), true) = (
                        actual.is_numeric(),
                        actual.has_concrete_skeleton(),
                        source,
                        found_candidate,
                    ) {
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

                                err.span_suggestion(
                                    lit.span,
                                    &format!(
                                        "you must specify a concrete type for \
                                              this numeric value, like `{}`",
                                        concrete_type
                                    ),
                                    format!("{}_{}", snippet, concrete_type),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            ExprKind::Path(ref qpath) => {
                                // local binding
                                if let QPath::Resolved(_, path) = qpath {
                                    if let hir::def::Res::Local(hir_id) = path.res {
                                        let span = tcx.hir().span(hir_id);
                                        let snippet = tcx.sess.source_map().span_to_snippet(span);
                                        let filename = tcx.sess.source_map().span_to_filename(span);

                                        let parent_node = self
                                            .tcx
                                            .hir()
                                            .get(self.tcx.hir().get_parent_node(hir_id));
                                        let msg = format!(
                                            "you must specify a type for this binding, like `{}`",
                                            concrete_type,
                                        );

                                        match (filename, parent_node, snippet) {
                                            (
                                                FileName::Real(_),
                                                Node::Local(hir::Local {
                                                    source: hir::LocalSource::Normal,
                                                    ty,
                                                    ..
                                                }),
                                                Ok(ref snippet),
                                            ) => {
                                                err.span_suggestion(
                                                    // account for `let x: _ = 42;`
                                                    //                  ^^^^
                                                    span.to(ty
                                                        .as_ref()
                                                        .map(|ty| ty.span)
                                                        .unwrap_or(span)),
                                                    &msg,
                                                    format!("{}: {}", snippet, concrete_type),
                                                    Applicability::MaybeIncorrect,
                                                );
                                            }
                                            _ => {
                                                err.span_label(span, msg);
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                        err.emit();
                        return None;
                    } else {
                        span = item_name.span;
                        let mut err = struct_span_err!(
                            tcx.sess,
                            span,
                            E0599,
                            "no {} named `{}` found for {} `{}` in the current scope",
                            item_kind,
                            item_name,
                            actual.prefix_string(),
                            ty_str,
                        );
                        if let Mode::MethodCall = mode {
                            if let SelfSource::MethodCall(call) = source {
                                self.suggest_await_before_method(
                                    &mut err, item_name, actual, call, span,
                                );
                            }
                        }
                        if let Some(span) =
                            tcx.sess.confused_type_with_std_module.borrow().get(&span)
                        {
                            if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(*span) {
                                err.span_suggestion(
                                    *span,
                                    "you are looking for the module in `std`, \
                                     not the primitive type",
                                    format!("std::{}", snippet),
                                    Applicability::MachineApplicable,
                                );
                            }
                        }
                        if let ty::RawPtr(_) = &actual.kind() {
                            err.note(
                                "try using `<*const T>::as_ref()` to get a reference to the \
                                      type behind the pointer: https://doc.rust-lang.org/std/\
                                      primitive.pointer.html#method.as_ref",
                            );
                            err.note(
                                "using `<*const T>::as_ref()` on a pointer \
                                      which is unaligned or points to invalid \
                                      or uninitialized memory is undefined behavior",
                            );
                        }
                        err
                    }
                } else {
                    tcx.sess.diagnostic().struct_dummy()
                };

                if let Some(def) = actual.ty_adt_def() {
                    if let Some(full_sp) = tcx.hir().span_if_local(def.did) {
                        let def_sp = tcx.sess.source_map().guess_head_span(full_sp);
                        err.span_label(
                            def_sp,
                            format!(
                                "{} `{}` not found {}",
                                item_kind,
                                item_name,
                                if def.is_enum() && !is_method { "here" } else { "for this" }
                            ),
                        );
                    }
                }

                // If the method name is the name of a field with a function or closure type,
                // give a helping note that it has to be called as `(x.f)(...)`.
                if let SelfSource::MethodCall(expr) = source {
                    let field_receiver =
                        self.autoderef(span, rcvr_ty).find_map(|(ty, _)| match ty.kind() {
                            ty::Adt(def, substs) if !def.is_enum() => {
                                let variant = &def.non_enum_variant();
                                self.tcx.find_field_index(item_name, variant).map(|index| {
                                    let field = &variant.fields[index];
                                    let field_ty = field.ty(tcx, substs);
                                    (field, field_ty)
                                })
                            }
                            _ => None,
                        });

                    if let Some((field, field_ty)) = field_receiver {
                        let scope = self.tcx.parent_module(self.body_id).to_def_id();
                        let is_accessible = field.vis.is_accessible_from(scope, self.tcx);

                        if is_accessible {
                            if self.is_fn_ty(&field_ty, span) {
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
                                let call_expr = self
                                    .tcx
                                    .hir()
                                    .expect_expr(self.tcx.hir().get_parent_node(expr.hir_id));

                                if let Some(span) = call_expr.span.trim_start(item_name.span) {
                                    err.span_suggestion(
                                        span,
                                        "remove the arguments",
                                        String::new(),
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            }
                        }

                        let field_kind = if is_accessible { "field" } else { "private field" };
                        err.span_label(item_name.span, format!("{}, not a method", field_kind));
                    } else if lev_candidate.is_none() && static_sources.is_empty() {
                        err.span_label(span, format!("{} not found in `{}`", item_kind, ty_str));
                        self.tcx.sess.trait_methods_not_found.borrow_mut().insert(orig_span);
                    }
                } else {
                    err.span_label(span, format!("{} not found in `{}`", item_kind, ty_str));
                    self.tcx.sess.trait_methods_not_found.borrow_mut().insert(orig_span);
                }

                if self.is_fn_ty(&rcvr_ty, span) {
                    macro_rules! report_function {
                        ($span:expr, $name:expr) => {
                            err.note(&format!(
                                "`{}` is a function, perhaps you wish to call it",
                                $name
                            ));
                        };
                    }

                    if let SelfSource::MethodCall(expr) = source {
                        if let Ok(expr_string) = tcx.sess.source_map().span_to_snippet(expr.span) {
                            report_function!(expr.span, expr_string);
                        } else if let ExprKind::Path(QPath::Resolved(_, ref path)) = expr.kind {
                            if let Some(segment) = path.segments.last() {
                                report_function!(expr.span, segment.ident);
                            }
                        }
                    }
                }

                if !static_sources.is_empty() {
                    err.note(
                        "found the following associated functions; to be used as methods, \
                         functions must have a `self` parameter",
                    );
                    err.span_label(span, "this is an associated function, not a method");
                }
                if static_sources.len() == 1 {
                    let ty_str = if let Some(CandidateSource::ImplSource(impl_did)) =
                        static_sources.get(0)
                    {
                        // When the "method" is resolved through dereferencing, we really want the
                        // original type that has the associated function for accurate suggestions.
                        // (#61411)
                        let ty = tcx.at(span).type_of(*impl_did);
                        match (&ty.peel_refs().kind(), &actual.peel_refs().kind()) {
                            (ty::Adt(def, _), ty::Adt(def_actual, _)) if def == def_actual => {
                                // Use `actual` as it will have more `substs` filled in.
                                self.ty_to_value_string(actual.peel_refs())
                            }
                            _ => self.ty_to_value_string(ty.peel_refs()),
                        }
                    } else {
                        self.ty_to_value_string(actual.peel_refs())
                    };
                    if let SelfSource::MethodCall(expr) = source {
                        err.span_suggestion(
                            expr.span.to(span),
                            "use associated function syntax instead",
                            format!("{}::{}", ty_str, item_name),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.help(&format!("try with `{}::{}`", ty_str, item_name,));
                    }

                    report_candidates(span, &mut err, static_sources, sugg_span);
                } else if static_sources.len() > 1 {
                    report_candidates(span, &mut err, static_sources, sugg_span);
                }

                let mut restrict_type_params = false;
                if !unsatisfied_predicates.is_empty() {
                    let def_span = |def_id| {
                        self.tcx.sess.source_map().guess_head_span(self.tcx.def_span(def_id))
                    };
                    let mut type_params = FxHashMap::default();
                    let mut bound_spans = vec![];

                    let mut collect_type_param_suggestions =
                        |self_ty: Ty<'tcx>, parent_pred: &ty::Predicate<'tcx>, obligation: &str| {
                            // We don't care about regions here, so it's fine to skip the binder here.
                            if let (ty::Param(_), ty::PredicateKind::Trait(p, _)) =
                                (self_ty.kind(), parent_pred.kind().skip_binder())
                            {
                                if let ty::Adt(def, _) = p.trait_ref.self_ty().kind() {
                                    let node = def.did.as_local().map(|def_id| {
                                        self.tcx
                                            .hir()
                                            .get(self.tcx.hir().local_def_id_to_hir_id(def_id))
                                    });
                                    if let Some(hir::Node::Item(hir::Item { kind, .. })) = node {
                                        if let Some(g) = kind.generics() {
                                            let key = match &g.where_clause.predicates[..] {
                                                [.., pred] => (pred.span().shrink_to_hi(), false),
                                                [] => (
                                                    g.where_clause
                                                        .span_for_predicates_or_empty_place(),
                                                    true,
                                                ),
                                            };
                                            type_params
                                                .entry(key)
                                                .or_insert_with(FxHashSet::default)
                                                .insert(obligation.to_owned());
                                        }
                                    }
                                }
                            }
                        };
                    let mut bound_span_label = |self_ty: Ty<'_>, obligation: &str, quiet: &str| {
                        let msg = format!(
                            "doesn't satisfy `{}`",
                            if obligation.len() > 50 { quiet } else { obligation }
                        );
                        match &self_ty.kind() {
                            // Point at the type that couldn't satisfy the bound.
                            ty::Adt(def, _) => bound_spans.push((def_span(def.did), msg)),
                            // Point at the trait object that couldn't satisfy the bound.
                            ty::Dynamic(preds, _) => {
                                for pred in preds.iter() {
                                    match pred.skip_binder() {
                                        ty::ExistentialPredicate::Trait(tr) => {
                                            bound_spans.push((def_span(tr.def_id), msg.clone()))
                                        }
                                        ty::ExistentialPredicate::Projection(_)
                                        | ty::ExistentialPredicate::AutoTrait(_) => {}
                                    }
                                }
                            }
                            // Point at the closure that couldn't satisfy the bound.
                            ty::Closure(def_id, _) => bound_spans
                                .push((def_span(*def_id), format!("doesn't satisfy `{}`", quiet))),
                            _ => {}
                        }
                    };
                    let mut format_pred = |pred: ty::Predicate<'tcx>| {
                        let bound_predicate = pred.kind();
                        match bound_predicate.skip_binder() {
                            ty::PredicateKind::Projection(pred) => {
                                let pred = bound_predicate.rebind(pred);
                                // `<Foo as Iterator>::Item = String`.
                                let trait_ref =
                                    pred.skip_binder().projection_ty.trait_ref(self.tcx);
                                let assoc = self
                                    .tcx
                                    .associated_item(pred.skip_binder().projection_ty.item_def_id);
                                let ty = pred.skip_binder().ty;
                                let obligation = format!("{}::{} = {}", trait_ref, assoc.ident, ty);
                                let quiet = format!(
                                    "<_ as {}>::{} = {}",
                                    trait_ref.print_only_trait_path(),
                                    assoc.ident,
                                    ty
                                );
                                bound_span_label(trait_ref.self_ty(), &obligation, &quiet);
                                Some((obligation, trait_ref.self_ty()))
                            }
                            ty::PredicateKind::Trait(poly_trait_ref, _) => {
                                let p = poly_trait_ref.trait_ref;
                                let self_ty = p.self_ty();
                                let path = p.print_only_trait_path();
                                let obligation = format!("{}: {}", self_ty, path);
                                let quiet = format!("_: {}", path);
                                bound_span_label(self_ty, &obligation, &quiet);
                                Some((obligation, self_ty))
                            }
                            _ => None,
                        }
                    };
                    let mut bound_list = unsatisfied_predicates
                        .iter()
                        .filter_map(|(pred, parent_pred)| {
                            format_pred(*pred).map(|(p, self_ty)| match parent_pred {
                                None => format!("`{}`", &p),
                                Some(parent_pred) => match format_pred(*parent_pred) {
                                    None => format!("`{}`", &p),
                                    Some((parent_p, _)) => {
                                        collect_type_param_suggestions(self_ty, parent_pred, &p);
                                        format!("`{}`\nwhich is required by `{}`", p, parent_p)
                                    }
                                },
                            })
                        })
                        .enumerate()
                        .collect::<Vec<(usize, String)>>();
                    for ((span, empty_where), obligations) in type_params.into_iter() {
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
                            format!(
                                "{} {}",
                                if empty_where { " where" } else { "," },
                                obligations.join(", ")
                            ),
                            Applicability::MaybeIncorrect,
                        );
                    }

                    bound_list.sort_by(|(_, a), (_, b)| a.cmp(&b)); // Sort alphabetically.
                    bound_list.dedup_by(|(_, a), (_, b)| a == b); // #35677
                    bound_list.sort_by_key(|(pos, _)| *pos); // Keep the original predicate order.
                    bound_spans.sort();
                    bound_spans.dedup();
                    for (span, msg) in bound_spans.into_iter() {
                        err.span_label(span, &msg);
                    }
                    if !bound_list.is_empty() {
                        let bound_list = bound_list
                            .into_iter()
                            .map(|(_, path)| path)
                            .collect::<Vec<_>>()
                            .join("\n");
                        err.note(&format!(
                            "the method `{}` exists but the following trait bounds were not \
                             satisfied:\n{}",
                            item_name, bound_list
                        ));
                    }
                }

                if actual.is_numeric() && actual.is_fresh() || restrict_type_params {
                } else {
                    self.suggest_traits_to_import(
                        &mut err,
                        span,
                        rcvr_ty,
                        item_name,
                        source,
                        out_of_scope_traits,
                        &unsatisfied_predicates,
                    );
                }

                if actual.is_enum() {
                    let adt_def = actual.ty_adt_def().expect("enum is not an ADT");
                    if let Some(suggestion) = lev_distance::find_best_match_for_name(
                        &adt_def.variants.iter().map(|s| s.ident.name).collect::<Vec<_>>(),
                        item_name.name,
                        None,
                    ) {
                        err.span_suggestion(
                            span,
                            "there is a variant with a similar name",
                            suggestion.to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }

                let mut fallback_span = true;
                let msg = "remove this method call";
                if item_name.name == sym::as_str && actual.peel_refs().is_str() {
                    if let SelfSource::MethodCall(expr) = source {
                        let call_expr =
                            self.tcx.hir().expect_expr(self.tcx.hir().get_parent_node(expr.hir_id));
                        if let Some(span) = call_expr.span.trim_start(expr.span) {
                            err.span_suggestion(
                                span,
                                msg,
                                String::new(),
                                Applicability::MachineApplicable,
                            );
                            fallback_span = false;
                        }
                    }
                    if fallback_span {
                        err.span_label(span, msg);
                    }
                } else if let Some(lev_candidate) = lev_candidate {
                    let def_kind = lev_candidate.kind.as_def_kind();
                    err.span_suggestion(
                        span,
                        &format!(
                            "there is {} {} with a similar name",
                            def_kind.article(),
                            def_kind.descr(lev_candidate.def_id),
                        ),
                        lev_candidate.ident.to_string(),
                        Applicability::MaybeIncorrect,
                    );
                }

                return Some(err);
            }

            MethodError::Ambiguity(sources) => {
                let mut err = struct_span_err!(
                    self.sess(),
                    item_name.span,
                    E0034,
                    "multiple applicable items in scope"
                );
                err.span_label(item_name.span, format!("multiple `{}` found", item_name));

                report_candidates(span, &mut err, sources, sugg_span);
                err.emit();
            }

            MethodError::PrivateMatch(kind, def_id, out_of_scope_traits) => {
                let kind = kind.descr(def_id);
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    item_name.span,
                    E0624,
                    "{} `{}` is private",
                    kind,
                    item_name
                );
                err.span_label(item_name.span, &format!("private {}", kind));
                self.suggest_valid_traits(&mut err, out_of_scope_traits);
                err.emit();
            }

            MethodError::IllegalSizedBound(candidates, needs_mut, bound_span) => {
                let msg = format!("the `{}` method cannot be invoked on a trait object", item_name);
                let mut err = self.sess().struct_span_err(span, &msg);
                err.span_label(bound_span, "this has a `Sized` requirement");
                if !candidates.is_empty() {
                    let help = format!(
                        "{an}other candidate{s} {were} found in the following trait{s}, perhaps \
                         add a `use` for {one_of_them}:",
                        an = if candidates.len() == 1 { "an" } else { "" },
                        s = pluralize!(candidates.len()),
                        were = if candidates.len() == 1 { "was" } else { "were" },
                        one_of_them = if candidates.len() == 1 { "it" } else { "one_of_them" },
                    );
                    self.suggest_use_candidates(&mut err, help, candidates);
                }
                if let ty::Ref(region, t_type, mutability) = rcvr_ty.kind() {
                    if needs_mut {
                        let trait_type = self.tcx.mk_ref(
                            region,
                            ty::TypeAndMut { ty: t_type, mutbl: mutability.invert() },
                        );
                        err.note(&format!("you need `{}` instead of `{}`", trait_type, rcvr_ty));
                    }
                }
                err.emit();
            }

            MethodError::BadReturnType => bug!("no return type expectations but got BadReturnType"),
        }
        None
    }

    /// Print out the type for use in value namespace.
    fn ty_to_value_string(&self, ty: Ty<'tcx>) -> String {
        match ty.kind() {
            ty::Adt(def, substs) => format!("{}", ty::Instance::new(def.did, substs)),
            _ => self.ty_to_string(ty),
        }
    }

    fn suggest_await_before_method(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        item_name: Ident,
        ty: Ty<'tcx>,
        call: &hir::Expr<'_>,
        span: Span,
    ) {
        let output_ty = match self.infcx.get_impl_future_output_ty(ty) {
            Some(output_ty) => self.resolve_vars_if_possible(output_ty),
            _ => return,
        };
        let method_exists = self.method_exists(item_name, output_ty, call.hir_id, true);
        debug!("suggest_await_before_method: is_method_exist={}", method_exists);
        if method_exists {
            err.span_suggestion_verbose(
                span.shrink_to_lo(),
                "consider `await`ing on the `Future` and calling the method on its `Output`",
                "await.".to_string(),
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn suggest_use_candidates(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        mut msg: String,
        candidates: Vec<DefId>,
    ) {
        let module_did = self.tcx.parent_module(self.body_id);
        let module_id = self.tcx.hir().local_def_id_to_hir_id(module_did);
        let krate = self.tcx.hir().krate();
        let (span, found_use) = UsePlacementFinder::check(self.tcx, krate, module_id);
        if let Some(span) = span {
            let path_strings = candidates.iter().map(|did| {
                // Produce an additional newline to separate the new use statement
                // from the directly following item.
                let additional_newline = if found_use { "" } else { "\n" };
                format!(
                    "use {};\n{}",
                    with_crate_prefix(|| self.tcx.def_path_str(*did)),
                    additional_newline
                )
            });

            err.span_suggestions(span, &msg, path_strings, Applicability::MaybeIncorrect);
        } else {
            let limit = if candidates.len() == 5 { 5 } else { 4 };
            for (i, trait_did) in candidates.iter().take(limit).enumerate() {
                if candidates.len() > 1 {
                    msg.push_str(&format!(
                        "\ncandidate #{}: `use {};`",
                        i + 1,
                        with_crate_prefix(|| self.tcx.def_path_str(*trait_did))
                    ));
                } else {
                    msg.push_str(&format!(
                        "\n`use {};`",
                        with_crate_prefix(|| self.tcx.def_path_str(*trait_did))
                    ));
                }
            }
            if candidates.len() > limit {
                msg.push_str(&format!("\nand {} others", candidates.len() - limit));
            }
            err.note(&msg[..]);
        }
    }

    fn suggest_valid_traits(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        valid_out_of_scope_traits: Vec<DefId>,
    ) -> bool {
        if !valid_out_of_scope_traits.is_empty() {
            let mut candidates = valid_out_of_scope_traits;
            candidates.sort();
            candidates.dedup();
            err.help("items from traits can only be used if the trait is in scope");
            let msg = format!(
                "the following {traits_are} implemented but not in scope; \
                 perhaps add a `use` for {one_of_them}:",
                traits_are = if candidates.len() == 1 { "trait is" } else { "traits are" },
                one_of_them = if candidates.len() == 1 { "it" } else { "one of them" },
            );

            self.suggest_use_candidates(err, msg, candidates);
            true
        } else {
            false
        }
    }

    fn suggest_traits_to_import<'b>(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        source: SelfSource<'b>,
        valid_out_of_scope_traits: Vec<DefId>,
        unsatisfied_predicates: &[(ty::Predicate<'tcx>, Option<ty::Predicate<'tcx>>)],
    ) {
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
                // We approximate the coherence rules to only suggest
                // traits that are legal to implement by requiring that
                // either the type or trait is local. Multi-dispatch means
                // this isn't perfect (that is, there are cases when
                // implementing a trait would be legal but is rejected
                // here).
                unsatisfied_predicates.iter().all(|(p, _)| {
                    match p.kind().skip_binder() {
                        // Hide traits if they are present in predicates as they can be fixed without
                        // having to implement them.
                        ty::PredicateKind::Trait(t, _) => t.def_id() == info.def_id,
                        ty::PredicateKind::Projection(p) => {
                            p.projection_ty.item_def_id == info.def_id
                        }
                        _ => false,
                    }
                }) && (type_is_local || info.def_id.is_local())
                    && self
                        .associated_item(info.def_id, item_name, Namespace::ValueNS)
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
                            item.vis == ty::Visibility::Public || info.def_id.is_local()
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

        if !candidates.is_empty() {
            // Sort from most relevant to least relevant.
            candidates.sort_by(|a, b| a.cmp(b).reverse());
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
            if let (Some(ref param), Some(ref table)) =
                (param_type, self.in_progress_typeck_results)
            {
                let table_owner = table.borrow().hir_owner;
                let generics = self.tcx.generics_of(table_owner.to_def_id());
                let type_param = generics.type_param(param, self.tcx);
                let hir = &self.tcx.hir();
                if let Some(def_id) = type_param.def_id.as_local() {
                    let id = hir.local_def_id_to_hir_id(def_id);
                    // Get the `hir::Param` to verify whether it already has any bounds.
                    // We do this to avoid suggesting code that ends up as `T: FooBar`,
                    // instead we suggest `T: Foo + Bar` in that case.
                    match hir.get(id) {
                        Node::GenericParam(ref param) => {
                            let mut impl_trait = false;
                            let has_bounds =
                                if let hir::GenericParamKind::Type { synthetic: Some(_), .. } =
                                    &param.kind
                                {
                                    // We've found `fn foo(x: impl Trait)` instead of
                                    // `fn foo<T>(x: T)`. We want to suggest the correct
                                    // `fn foo(x: impl Trait + TraitBound)` instead of
                                    // `fn foo<T: TraitBound>(x: T)`. (#63706)
                                    impl_trait = true;
                                    param.bounds.get(1)
                                } else {
                                    param.bounds.get(0)
                                };
                            let sp = hir.span(id);
                            let sp = if let Some(first_bound) = has_bounds {
                                // `sp` only covers `T`, change it so that it covers
                                // `T:` when appropriate
                                sp.until(first_bound.span())
                            } else {
                                sp
                            };
                            let trait_def_ids: FxHashSet<DefId> = param
                                .bounds
                                .iter()
                                .filter_map(|bound| Some(bound.trait_ref()?.trait_def_id()?))
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
                                            "{}{} {}{}",
                                            param.name.ident(),
                                            if impl_trait { " +" } else { ":" },
                                            self.tcx.def_path_str(t.def_id),
                                            if has_bounds.is_some() { " + " } else { "" },
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
            } else if let Some(simp_rcvr_ty) = simplify_type(self.tcx, rcvr_ty, true) {
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
                            let imp = self.tcx.impl_trait_ref(imp_did).unwrap();
                            let imp_simp = simplify_type(self.tcx, imp.self_ty(), true);
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
                    let span = self.tcx.hir().span_if_local(trait_info.def_id).unwrap();
                    err.span_note(
                        self.tcx.sess.source_map().guess_head_span(span),
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
                        "the trait `{}` defines an item `{}`, but is explicitely unimplemented",
                        self.tcx.def_path_str(trait_info.def_id),
                        item_name
                    );
                    err.note(&msg);
                }
                trait_infos => {
                    let mut msg = format!(
                        "the following traits define an item `{}`, but are explicitely unimplemented:",
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

    /// Checks whether there is a local type somewhere in the chain of
    /// autoderefs of `rcvr_ty`.
    fn type_derefs_to_local(&self, span: Span, rcvr_ty: Ty<'tcx>, source: SelfSource<'_>) -> bool {
        fn is_local(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Adt(def, _) => def.did.is_local(),
                ty::Foreign(did) => did.is_local(),
                ty::Dynamic(ref tr, ..) => tr.principal().map_or(false, |d| d.def_id().is_local()),
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

#[derive(Copy, Clone)]
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

/// Retrieves all traits in this crate and any dependent crates.
pub fn all_traits(tcx: TyCtxt<'_>) -> Vec<TraitInfo> {
    tcx.all_traits(LOCAL_CRATE).iter().map(|&def_id| TraitInfo { def_id }).collect()
}

/// Computes all traits in this crate and any dependent crates.
fn compute_all_traits(tcx: TyCtxt<'_>) -> Vec<DefId> {
    use hir::itemlikevisit;

    let mut traits = vec![];

    // Crate-local:

    struct Visitor<'a, 'tcx> {
        map: &'a hir_map::Map<'tcx>,
        traits: &'a mut Vec<DefId>,
    }

    impl<'v, 'a, 'tcx> itemlikevisit::ItemLikeVisitor<'v> for Visitor<'a, 'tcx> {
        fn visit_item(&mut self, i: &'v hir::Item<'v>) {
            match i.kind {
                hir::ItemKind::Trait(..) | hir::ItemKind::TraitAlias(..) => {
                    let def_id = self.map.local_def_id(i.hir_id);
                    self.traits.push(def_id.to_def_id());
                }
                _ => (),
            }
        }

        fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

        fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

        fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
    }

    tcx.hir().krate().visit_all_item_likes(&mut Visitor { map: &tcx.hir(), traits: &mut traits });

    // Cross-crate:

    let mut external_mods = FxHashSet::default();
    fn handle_external_res(
        tcx: TyCtxt<'_>,
        traits: &mut Vec<DefId>,
        external_mods: &mut FxHashSet<DefId>,
        res: Res,
    ) {
        match res {
            Res::Def(DefKind::Trait | DefKind::TraitAlias, def_id) => {
                traits.push(def_id);
            }
            Res::Def(DefKind::Mod, def_id) => {
                if !external_mods.insert(def_id) {
                    return;
                }
                for child in tcx.item_children(def_id).iter() {
                    handle_external_res(tcx, traits, external_mods, child.res)
                }
            }
            _ => {}
        }
    }
    for &cnum in tcx.crates().iter() {
        let def_id = DefId { krate: cnum, index: CRATE_DEF_INDEX };
        handle_external_res(tcx, &mut traits, &mut external_mods, Res::Def(DefKind::Mod, def_id));
    }

    traits
}

pub fn provide(providers: &mut ty::query::Providers) {
    providers.all_traits = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        &tcx.arena.alloc(compute_all_traits(tcx))[..]
    }
}

struct UsePlacementFinder<'tcx> {
    target_module: hir::HirId,
    span: Option<Span>,
    found_use: bool,
    tcx: TyCtxt<'tcx>,
}

impl UsePlacementFinder<'tcx> {
    fn check(
        tcx: TyCtxt<'tcx>,
        krate: &'tcx hir::Crate<'tcx>,
        target_module: hir::HirId,
    ) -> (Option<Span>, bool) {
        let mut finder = UsePlacementFinder { target_module, span: None, found_use: false, tcx };
        intravisit::walk_crate(&mut finder, krate);
        (finder.span, finder.found_use)
    }
}

impl intravisit::Visitor<'tcx> for UsePlacementFinder<'tcx> {
    fn visit_mod(&mut self, module: &'tcx hir::Mod<'tcx>, _: Span, hir_id: hir::HirId) {
        if self.span.is_some() {
            return;
        }
        if hir_id != self.target_module {
            intravisit::walk_mod(self, module, hir_id);
            return;
        }
        // Find a `use` statement.
        for item_id in module.item_ids {
            let item = self.tcx.hir().expect_item(item_id.id);
            match item.kind {
                hir::ItemKind::Use(..) => {
                    // Don't suggest placing a `use` before the prelude
                    // import or other generated ones.
                    if !item.span.from_expansion() {
                        self.span = Some(item.span.shrink_to_lo());
                        self.found_use = true;
                        return;
                    }
                }
                // Don't place `use` before `extern crate`...
                hir::ItemKind::ExternCrate(_) => {}
                // ...but do place them before the first other item.
                _ => {
                    if self.span.map_or(true, |span| item.span < span) {
                        if !item.span.from_expansion() {
                            // Don't insert between attributes and an item.
                            if item.attrs.is_empty() {
                                self.span = Some(item.span.shrink_to_lo());
                            } else {
                                // Find the first attribute on the item.
                                for attr in item.attrs {
                                    if self.span.map_or(true, |span| attr.span < span) {
                                        self.span = Some(attr.span.shrink_to_lo());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::None
    }
}

fn print_disambiguation_help(
    item_name: Ident,
    args: Option<&'tcx [hir::Expr<'tcx>]>,
    err: &mut DiagnosticBuilder<'_>,
    trait_name: String,
    rcvr_ty: Ty<'_>,
    kind: ty::AssocKind,
    def_id: DefId,
    span: Span,
    candidate: Option<usize>,
    source_map: &source_map::SourceMap,
) {
    let mut applicability = Applicability::MachineApplicable;
    let sugg_args = if let (ty::AssocKind::Fn, Some(args)) = (kind, args) {
        format!(
            "({}{})",
            if rcvr_ty.is_region_ptr() {
                if rcvr_ty.is_mutable_ptr() { "&mut " } else { "&" }
            } else {
                ""
            },
            args.iter()
                .map(|arg| source_map.span_to_snippet(arg.span).unwrap_or_else(|_| {
                    applicability = Applicability::HasPlaceholders;
                    "_".to_owned()
                }))
                .collect::<Vec<_>>()
                .join(", "),
        )
    } else {
        String::new()
    };
    let sugg = format!("{}::{}{}", trait_name, item_name, sugg_args);
    err.span_suggestion(
        span,
        &format!(
            "disambiguate the {} for {}",
            kind.as_def_kind().descr(def_id),
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

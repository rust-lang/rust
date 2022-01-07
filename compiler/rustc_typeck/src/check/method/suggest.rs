//! Give useful errors and suggestions to users when an item can't be
//! found or is otherwise invalid.

use crate::check::FnCtxt;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::Namespace;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ExprKind, Node, QPath};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::ty::fast_reject::{simplify_type, SimplifyParams, StripReferences};
use rustc_middle::ty::print::with_crate_prefix;
use rustc_middle::ty::{self, DefIdTree, ToPredicate, Ty, TyCtxt, TypeFoldable};
use rustc_span::lev_distance;
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::{source_map, FileName, MultiSpan, Span, Symbol};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{
    FulfillmentError, Obligation, ObligationCause, ObligationCauseCode,
};

use std::cmp::Ordering;
use std::iter;

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
                        let poly_trait_ref = ty::Binder::dummy(trait_ref);
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

    fn is_slice_ty(&self, ty: Ty<'tcx>, span: Span) -> bool {
        self.autoderef(span, ty).any(|(ty, _)| matches!(ty.kind(), ty::Slice(..) | ty::Array(..)))
    }

    pub fn report_method_error(
        &self,
        mut span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        source: SelfSource<'tcx>,
        error: MethodError<'tcx>,
        args: Option<&'tcx [hir::Expr<'tcx>]>,
    ) -> Option<DiagnosticBuilder<'_>> {
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
                                item.fn_has_self_parameter,
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
                            item.fn_has_self_parameter,
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
                                    let snippet = tcx.sess.source_map().span_to_snippet(span);
                                    let filename = tcx.sess.source_map().span_to_filename(span);

                                    let parent_node =
                                        self.tcx.hir().get(self.tcx.hir().get_parent_node(hir_id));
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
                            _ => {}
                        }
                        err.emit();
                        return None;
                    } else {
                        span = item_name.span;

                        // Don't show generic arguments when the method can't be found in any implementation (#81576).
                        let mut ty_str_reported = ty_str.clone();
                        if let ty::Adt(_, generics) = actual.kind() {
                            if generics.len() > 0 {
                                let mut autoderef = self.autoderef(span, actual);
                                let candidate_found = autoderef.any(|(ty, _)| {
                                    if let ty::Adt(adt_deref, _) = ty.kind() {
                                        self.tcx
                                            .inherent_impls(adt_deref.did)
                                            .iter()
                                            .filter_map(|def_id| {
                                                self.associated_item(
                                                    *def_id,
                                                    item_name,
                                                    Namespace::ValueNS,
                                                )
                                            })
                                            .count()
                                            >= 1
                                    } else {
                                        false
                                    }
                                });
                                let has_deref = autoderef.step_count() > 0;
                                if !candidate_found
                                    && !has_deref
                                    && unsatisfied_predicates.is_empty()
                                {
                                    if let Some((path_string, _)) = ty_str.split_once('<') {
                                        ty_str_reported = path_string.to_string();
                                    }
                                }
                            }
                        }

                        let mut err = struct_span_err!(
                            tcx.sess,
                            span,
                            E0599,
                            "no {} named `{}` found for {} `{}` in the current scope",
                            item_kind,
                            item_name,
                            actual.prefix_string(self.tcx),
                            ty_str_reported,
                        );
                        if let Mode::MethodCall = mode {
                            if let SelfSource::MethodCall(call) = source {
                                self.suggest_await_before_method(
                                    &mut err, item_name, actual, call, span,
                                );
                            }
                        }
                        if let Some(span) =
                            tcx.resolutions(()).confused_type_with_std_module.get(&span)
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

                let mut label_span_not_found = || {
                    if unsatisfied_predicates.is_empty() {
                        err.span_label(span, format!("{item_kind} not found in `{ty_str}`"));
                        let is_string_or_ref_str = match actual.kind() {
                            ty::Ref(_, ty, _) => {
                                ty.is_str()
                                    || matches!(
                                        ty.kind(),
                                        ty::Adt(adt, _) if self.tcx.is_diagnostic_item(sym::String, adt.did)
                                    )
                            }
                            ty::Adt(adt, _) => self.tcx.is_diagnostic_item(sym::String, adt.did),
                            _ => false,
                        };
                        if is_string_or_ref_str && item_name.name == sym::iter {
                            err.span_suggestion_verbose(
                                item_name.span,
                                "because of the in-memory representation of `&str`, to obtain \
                                 an `Iterator` over each of its codepoint use method `chars`",
                                String::from("chars"),
                                Applicability::MachineApplicable,
                            );
                        }
                        if let ty::Adt(adt, _) = rcvr_ty.kind() {
                            let mut inherent_impls_candidate = self
                                .tcx
                                .inherent_impls(adt.did)
                                .iter()
                                .copied()
                                .filter(|def_id| {
                                    if let Some(assoc) =
                                        self.associated_item(*def_id, item_name, Namespace::ValueNS)
                                    {
                                        // Check for both mode is the same so we avoid suggesting
                                        // incorrect associated item.
                                        match (mode, assoc.fn_has_self_parameter, source) {
                                            (Mode::MethodCall, true, SelfSource::MethodCall(_)) => {
                                                // We check that the suggest type is actually
                                                // different from the received one
                                                // So we avoid suggestion method with Box<Self>
                                                // for instance
                                                self.tcx.at(span).type_of(*def_id) != actual
                                                    && self.tcx.at(span).type_of(*def_id) != rcvr_ty
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
                                        format!("- `{}`", self.tcx.at(span).type_of(*impl_item))
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                let additional_types = if inherent_impls_candidate.len() > limit {
                                    format!(
                                        "\nand {} more types",
                                        inherent_impls_candidate.len() - limit
                                    )
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
                        err.span_label(span, format!("{item_kind} cannot be called on `{ty_str}` due to unsatisfied trait bounds"));
                    }
                };

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
                        label_span_not_found();
                    }
                } else {
                    label_span_not_found();
                }

                if self.is_fn_ty(rcvr_ty, span) {
                    fn report_function<T: std::fmt::Display>(
                        err: &mut DiagnosticBuilder<'_>,
                        name: T,
                    ) {
                        err.note(
                            &format!("`{}` is a function, perhaps you wish to call it", name,),
                        );
                    }

                    if let SelfSource::MethodCall(expr) = source {
                        if let Ok(expr_string) = tcx.sess.source_map().span_to_snippet(expr.span) {
                            report_function(&mut err, expr_string);
                        } else if let ExprKind::Path(QPath::Resolved(_, path)) = expr.kind {
                            if let Some(segment) = path.segments.last() {
                                report_function(&mut err, segment.ident);
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
                let mut unsatisfied_bounds = false;
                if item_name.name == sym::count && self.is_slice_ty(actual, span) {
                    let msg = "consider using `len` instead";
                    if let SelfSource::MethodCall(_expr) = source {
                        err.span_suggestion_short(
                            span,
                            msg,
                            String::from("len"),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.span_label(span, msg);
                    }
                    if let Some(iterator_trait) = self.tcx.get_diagnostic_item(sym::Iterator) {
                        let iterator_trait = self.tcx.def_path_str(iterator_trait);
                        err.note(&format!("`count` is defined on `{iterator_trait}`, which `{actual}` does not implement"));
                    }
                } else if !unsatisfied_predicates.is_empty() {
                    let def_span = |def_id| {
                        self.tcx.sess.source_map().guess_head_span(self.tcx.def_span(def_id))
                    };
                    let mut type_params = FxHashMap::default();
                    let mut bound_spans = vec![];

                    let mut collect_type_param_suggestions =
                        |self_ty: Ty<'tcx>, parent_pred: &ty::Predicate<'tcx>, obligation: &str| {
                            // We don't care about regions here, so it's fine to skip the binder here.
                            if let (ty::Param(_), ty::PredicateKind::Trait(p)) =
                                (self_ty.kind(), parent_pred.kind().skip_binder())
                            {
                                let node = match p.trait_ref.self_ty().kind() {
                                    ty::Param(_) => {
                                        // Account for `fn` items like in `issue-35677.rs` to
                                        // suggest restricting its type params.
                                        let did = self.tcx.hir().body_owner_def_id(hir::BodyId {
                                            hir_id: self.body_id,
                                        });
                                        Some(
                                            self.tcx
                                                .hir()
                                                .get(self.tcx.hir().local_def_id_to_hir_id(did)),
                                        )
                                    }
                                    ty::Adt(def, _) => def.did.as_local().map(|def_id| {
                                        self.tcx
                                            .hir()
                                            .get(self.tcx.hir().local_def_id_to_hir_id(def_id))
                                    }),
                                    _ => None,
                                };
                                if let Some(hir::Node::Item(hir::Item { kind, .. })) = node {
                                    if let Some(g) = kind.generics() {
                                        let key = match g.where_clause.predicates {
                                            [.., pred] => (pred.span().shrink_to_hi(), false),
                                            [] => (
                                                g.where_clause.span_for_predicates_or_empty_place(),
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
                                let projection_ty = pred.skip_binder().projection_ty;

                                let substs_with_infer_self = tcx.mk_substs(
                                    iter::once(tcx.mk_ty_var(ty::TyVid::from_u32(0)).into())
                                        .chain(projection_ty.substs.iter().skip(1)),
                                );

                                let quiet_projection_ty = ty::ProjectionTy {
                                    substs: substs_with_infer_self,
                                    item_def_id: projection_ty.item_def_id,
                                };

                                let ty = pred.skip_binder().ty;

                                let obligation = format!("{} = {}", projection_ty, ty);
                                let quiet = format!("{} = {}", quiet_projection_ty, ty);

                                bound_span_label(projection_ty.self_ty(), &obligation, &quiet);
                                Some((obligation, projection_ty.self_ty()))
                            }
                            ty::PredicateKind::Trait(poly_trait_ref) => {
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

                    // Find all the requirements that come from a local `impl` block.
                    let mut skip_list: FxHashSet<_> = Default::default();
                    let mut spanned_predicates: FxHashMap<MultiSpan, _> = Default::default();
                    for (data, p, parent_p) in unsatisfied_predicates
                        .iter()
                        .filter_map(|(p, parent, c)| c.as_ref().map(|c| (p, parent, c)))
                        .filter_map(|(p, parent, c)| match c.code() {
                            ObligationCauseCode::ImplDerivedObligation(ref data) => {
                                Some((data, p, parent))
                            }
                            _ => None,
                        })
                    {
                        let parent_trait_ref = data.parent_trait_ref;
                        let parent_def_id = parent_trait_ref.def_id();
                        let path = parent_trait_ref.print_only_trait_path();
                        let tr_self_ty = parent_trait_ref.skip_binder().self_ty();
                        let mut candidates = vec![];
                        self.tcx.for_each_relevant_impl(
                            parent_def_id,
                            parent_trait_ref.self_ty().skip_binder(),
                            |impl_def_id| match self.tcx.hir().get_if_local(impl_def_id) {
                                Some(Node::Item(hir::Item {
                                    kind: hir::ItemKind::Impl(hir::Impl { .. }),
                                    ..
                                })) => {
                                    candidates.push(impl_def_id);
                                }
                                _ => {}
                            },
                        );
                        if let [def_id] = &candidates[..] {
                            match self.tcx.hir().get_if_local(*def_id) {
                                Some(Node::Item(hir::Item {
                                    kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, .. }),
                                    ..
                                })) => {
                                    if let Some(pred) = parent_p {
                                        // Done to add the "doesn't satisfy" `span_label`.
                                        let _ = format_pred(*pred);
                                    }
                                    skip_list.insert(p);
                                    let mut spans = Vec::with_capacity(2);
                                    if let Some(trait_ref) = of_trait {
                                        spans.push(trait_ref.path.span);
                                    }
                                    spans.push(self_ty.span);
                                    let entry = spanned_predicates.entry(spans.into());
                                    entry
                                        .or_insert_with(|| (path, tr_self_ty, Vec::new()))
                                        .2
                                        .push(p);
                                }
                                _ => {}
                            }
                        }
                    }
                    for (span, (path, self_ty, preds)) in spanned_predicates {
                        err.span_note(
                            span,
                            &format!(
                                "the following trait bounds were not satisfied because of the \
                                 requirements of the implementation of `{}` for `{}`:\n{}",
                                path,
                                self_ty,
                                preds
                                    .into_iter()
                                    // .map(|pred| format!("{:?}", pred))
                                    .filter_map(|pred| format_pred(*pred))
                                    .map(|(p, _)| format!("`{}`", p))
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            ),
                        );
                    }

                    // The requirements that didn't have an `impl` span to show.
                    let mut bound_list = unsatisfied_predicates
                        .iter()
                        .filter(|(pred, _, _parent_pred)| !skip_list.contains(&pred))
                        .filter_map(|(pred, parent_pred, _cause)| {
                            format_pred(*pred).map(|(p, self_ty)| {
                                collect_type_param_suggestions(self_ty, pred, &p);
                                match parent_pred {
                                    None => format!("`{}`", &p),
                                    Some(parent_pred) => match format_pred(*parent_pred) {
                                        None => format!("`{}`", &p),
                                        Some((parent_p, _)) => {
                                            collect_type_param_suggestions(
                                                self_ty,
                                                parent_pred,
                                                &p,
                                            );
                                            format!("`{}`\nwhich is required by `{}`", p, parent_p)
                                        }
                                    },
                                }
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

                    bound_list.sort_by(|(_, a), (_, b)| a.cmp(b)); // Sort alphabetically.
                    bound_list.dedup_by(|(_, a), (_, b)| a == b); // #35677
                    bound_list.sort_by_key(|(pos, _)| *pos); // Keep the original predicate order.
                    bound_spans.sort();
                    bound_spans.dedup();
                    for (span, msg) in bound_spans.into_iter() {
                        err.span_label(span, &msg);
                    }
                    if !bound_list.is_empty() || !skip_list.is_empty() {
                        let bound_list = bound_list
                            .into_iter()
                            .map(|(_, path)| path)
                            .collect::<Vec<_>>()
                            .join("\n");
                        let actual_prefix = actual.prefix_string(self.tcx);
                        err.set_primary_message(&format!(
                            "the {item_kind} `{item_name}` exists for {actual_prefix} `{ty_str}`, but its trait bounds were not satisfied"
                        ));
                        if !bound_list.is_empty() {
                            err.note(&format!(
                                "the following trait bounds were not satisfied:\n{bound_list}"
                            ));
                        }
                        self.suggest_derive(&mut err, &unsatisfied_predicates);

                        unsatisfied_bounds = true;
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
                        unsatisfied_bounds,
                    );
                }

                // Don't emit a suggestion if we found an actual method
                // that had unsatisfied trait bounds
                if unsatisfied_predicates.is_empty() && actual.is_enum() {
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

                if item_name.name == sym::as_str && actual.peel_refs().is_str() {
                    let msg = "remove this method call";
                    let mut fallback_span = true;
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
                    // Don't emit a suggestion if we found an actual method
                    // that had unsatisfied trait bounds
                    if unsatisfied_predicates.is_empty() {
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
                let sp = self
                    .tcx
                    .hir()
                    .span_if_local(def_id)
                    .unwrap_or_else(|| self.tcx.def_span(def_id));
                err.span_label(sp, &format!("private {} defined here", kind));
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

    crate fn note_unmet_impls_on_type(
        &self,
        err: &mut rustc_errors::DiagnosticBuilder<'_>,
        errors: Vec<FulfillmentError<'tcx>>,
    ) {
        let all_local_types_needing_impls =
            errors.iter().all(|e| match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Trait(pred) => match pred.self_ty().kind() {
                    ty::Adt(def, _) => def.did.is_local(),
                    _ => false,
                },
                _ => false,
            });
        let mut preds: Vec<_> = errors
            .iter()
            .filter_map(|e| match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Trait(pred) => Some(pred),
                _ => None,
            })
            .collect();
        preds.sort_by_key(|pred| (pred.def_id(), pred.self_ty()));
        let def_ids = preds
            .iter()
            .filter_map(|pred| match pred.self_ty().kind() {
                ty::Adt(def, _) => Some(def.did),
                _ => None,
            })
            .collect::<FxHashSet<_>>();
        let sm = self.tcx.sess.source_map();
        let mut spans: MultiSpan = def_ids
            .iter()
            .filter_map(|def_id| {
                let span = self.tcx.def_span(*def_id);
                if span.is_dummy() { None } else { Some(sm.guess_head_span(span)) }
            })
            .collect::<Vec<_>>()
            .into();

        for pred in &preds {
            match pred.self_ty().kind() {
                ty::Adt(def, _) => {
                    spans.push_span_label(
                        sm.guess_head_span(self.tcx.def_span(def.did)),
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

    fn suggest_derive(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
    ) {
        let mut derives = Vec::<(String, Span, String)>::new();
        let mut traits = Vec::<Span>::new();
        for (pred, _, _) in unsatisfied_predicates {
            let trait_pred = match pred.kind().skip_binder() {
                ty::PredicateKind::Trait(trait_pred) => trait_pred,
                _ => continue,
            };
            let adt = match trait_pred.self_ty().ty_adt_def() {
                Some(adt) if adt.did.is_local() => adt,
                _ => continue,
            };
            let can_derive = match self.tcx.get_diagnostic_name(trait_pred.def_id()) {
                Some(sym::Default) => !adt.is_enum(),
                Some(
                    sym::Eq
                    | sym::PartialEq
                    | sym::Ord
                    | sym::PartialOrd
                    | sym::Clone
                    | sym::Copy
                    | sym::Hash
                    | sym::Debug,
                ) => true,
                _ => false,
            };
            if can_derive {
                derives.push((
                    format!("{}", trait_pred.self_ty()),
                    self.tcx.def_span(adt.did),
                    format!("{}", trait_pred.trait_ref.print_only_trait_name()),
                ));
            } else {
                traits.push(self.tcx.def_span(trait_pred.def_id()));
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
            derives_grouped.push((self_name, self_span, trait_name));
        }

        let len = traits.len();
        if len > 0 {
            let span: MultiSpan = traits.into();
            err.span_note(
                span,
                &format!("the following trait{} must be implemented", pluralize!(len),),
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
        let parent_map = self.tcx.visible_parent_map(());

        // Separate out candidates that must be imported with a glob, because they are named `_`
        // and cannot be referred with their identifier.
        let (candidates, globs): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|trait_did| {
            if let Some(parent_did) = parent_map.get(trait_did) {
                // If the item is re-exported as `_`, we should suggest a glob-import instead.
                if Some(*parent_did) != self.tcx.parent(*trait_did)
                    && self
                        .tcx
                        .item_children(*parent_did)
                        .iter()
                        .filter(|child| child.res.opt_def_id() == Some(*trait_did))
                        .all(|child| child.ident.name == kw::Underscore)
                {
                    return false;
                }
            }

            true
        });

        let module_did = self.tcx.parent_module(self.body_id);
        let (span, found_use) = find_use_placement(self.tcx, module_did);
        if let Some(span) = span {
            let path_strings = candidates.iter().map(|trait_did| {
                // Produce an additional newline to separate the new use statement
                // from the directly following item.
                let additional_newline = if found_use { "" } else { "\n" };
                format!(
                    "use {};\n{}",
                    with_crate_prefix(|| self.tcx.def_path_str(*trait_did)),
                    additional_newline
                )
            });

            let glob_path_strings = globs.iter().map(|trait_did| {
                let parent_did = parent_map.get(trait_did).unwrap();

                // Produce an additional newline to separate the new use statement
                // from the directly following item.
                let additional_newline = if found_use { "" } else { "\n" };
                format!(
                    "use {}::*; // trait {}\n{}",
                    with_crate_prefix(|| self.tcx.def_path_str(*parent_did)),
                    self.tcx.item_name(*trait_did),
                    additional_newline
                )
            });

            err.span_suggestions(
                span,
                &msg,
                path_strings.chain(glob_path_strings),
                Applicability::MaybeIncorrect,
            );
        } else {
            let limit = if candidates.len() + globs.len() == 5 { 5 } else { 4 };
            for (i, trait_did) in candidates.iter().take(limit).enumerate() {
                if candidates.len() + globs.len() > 1 {
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
            for (i, trait_did) in
                globs.iter().take(limit.saturating_sub(candidates.len())).enumerate()
            {
                let parent_did = parent_map.get(trait_did).unwrap();

                if candidates.len() + globs.len() > 1 {
                    msg.push_str(&format!(
                        "\ncandidate #{}: `use {}::*; // trait {}`",
                        candidates.len() + i + 1,
                        with_crate_prefix(|| self.tcx.def_path_str(*parent_did)),
                        self.tcx.item_name(*trait_did),
                    ));
                } else {
                    msg.push_str(&format!(
                        "\n`use {}::*; // trait {}`",
                        with_crate_prefix(|| self.tcx.def_path_str(*parent_did)),
                        self.tcx.item_name(*trait_did),
                    ));
                }
            }
            if candidates.len() > limit {
                msg.push_str(&format!("\nand {} others", candidates.len() + globs.len() - limit));
            }
            err.note(&msg);
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
                    with_crate_prefix(|| self.tcx.def_path_str(did))
                ));
            }

            true
        } else {
            false
        }
    }

    fn suggest_traits_to_import(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        source: SelfSource<'tcx>,
        valid_out_of_scope_traits: Vec<DefId>,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
        unsatisfied_bounds: bool,
    ) {
        let mut alt_rcvr_sugg = false;
        if let (SelfSource::MethodCall(rcvr), false) = (source, unsatisfied_bounds) {
            debug!(?span, ?item_name, ?rcvr_ty, ?rcvr);
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
                (self.tcx.mk_mut_ref(&ty::ReErased, rcvr_ty), "&mut "),
                (self.tcx.mk_imm_ref(&ty::ReErased, rcvr_ty), "&"),
            ] {
                if let Ok(pick) = self.lookup_probe(
                    span,
                    item_name,
                    rcvr_ty,
                    rcvr,
                    crate::check::method::probe::ProbeScope::AllTraits,
                ) {
                    // If the method is defined for the receiver we have, it likely wasn't `use`d.
                    // We point at the method, but we just skip the rest of the check for arbitrary
                    // self types and rely on the suggestion to `use` the trait from
                    // `suggest_valid_traits`.
                    let did = Some(pick.item.container.id());
                    let skip = skippable.contains(&did);
                    if pick.autoderefs == 0 && !skip {
                        err.span_label(
                            pick.item.ident.span,
                            &format!("the method is available for `{}` here", rcvr_ty),
                        );
                    }
                    break;
                }
                for (rcvr_ty, pre) in &[
                    (self.tcx.mk_lang_item(rcvr_ty, LangItem::OwnedBox), "Box::new"),
                    (self.tcx.mk_lang_item(rcvr_ty, LangItem::Pin), "Pin::new"),
                    (self.tcx.mk_diagnostic_item(rcvr_ty, sym::Arc), "Arc::new"),
                    (self.tcx.mk_diagnostic_item(rcvr_ty, sym::Rc), "Rc::new"),
                ] {
                    if let Some(new_rcvr_t) = *rcvr_ty {
                        if let Ok(pick) = self.lookup_probe(
                            span,
                            item_name,
                            new_rcvr_t,
                            rcvr,
                            crate::check::method::probe::ProbeScope::AllTraits,
                        ) {
                            debug!("try_alt_rcvr: pick candidate {:?}", pick);
                            let did = Some(pick.item.container.id());
                            // We don't want to suggest a container type when the missing
                            // method is `.clone()` or `.deref()` otherwise we'd suggest
                            // `Arc::new(foo).clone()`, which is far from what the user wants.
                            // Explicitly ignore the `Pin::as_ref()` method as `Pin` does not
                            // implement the `AsRef` trait.
                            let skip = skippable.contains(&did)
                                || (("Pin::new" == *pre)
                                    && (Symbol::intern("as_ref") == item_name.name));
                            // Make sure the method is defined for the *actual* receiver: we don't
                            // want to treat `Box<Self>` as a receiver if it only works because of
                            // an autoderef to `&self`
                            if pick.autoderefs == 0 && !skip {
                                err.span_label(
                                    pick.item.ident.span,
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
                        ty::PredicateKind::Trait(t) => t.def_id() == info.def_id,
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
                            item.vis.is_public() || info.def_id.is_local()
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
            if let (Some(param), Some(table)) = (param_type, self.in_progress_typeck_results) {
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
                        Node::GenericParam(param) => {
                            let mut impl_trait = false;
                            let has_bounds =
                                if let hir::GenericParamKind::Type { synthetic: true, .. } =
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
            } else if let Some(simp_rcvr_ty) =
                simplify_type(self.tcx, rcvr_ty, SimplifyParams::Yes, StripReferences::No)
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
                            let imp = self.tcx.impl_trait_ref(imp_did).unwrap();
                            let imp_simp = simplify_type(
                                self.tcx,
                                imp.self_ty(),
                                SimplifyParams::Yes,
                                StripReferences::No,
                            );
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
    fn type_derefs_to_local(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        source: SelfSource<'tcx>,
    ) -> bool {
        fn is_local(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Adt(def, _) => def.did.is_local(),
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

fn find_use_placement<'tcx>(tcx: TyCtxt<'tcx>, target_module: LocalDefId) -> (Option<Span>, bool) {
    let mut span = None;
    let mut found_use = false;
    let (module, _, _) = tcx.hir().get_module(target_module);

    // Find a `use` statement.
    for &item_id in module.item_ids {
        let item = tcx.hir().item(item_id);
        match item.kind {
            hir::ItemKind::Use(..) => {
                // Don't suggest placing a `use` before the prelude
                // import or other generated ones.
                if !item.span.from_expansion() {
                    span = Some(item.span.shrink_to_lo());
                    found_use = true;
                    break;
                }
            }
            // Don't place `use` before `extern crate`...
            hir::ItemKind::ExternCrate(_) => {}
            // ...but do place them before the first other item.
            _ => {
                if span.map_or(true, |span| item.span < span) {
                    if !item.span.from_expansion() {
                        span = Some(item.span.shrink_to_lo());
                        // Don't insert between attributes and an item.
                        let attrs = tcx.hir().attrs(item.hir_id());
                        // Find the first attribute on the item.
                        // FIXME: This is broken for active attributes.
                        for attr in attrs {
                            if !attr.span.is_dummy() && span.map_or(true, |span| attr.span < span) {
                                span = Some(attr.span.shrink_to_lo());
                            }
                        }
                    }
                }
            }
        }
    }

    (span, found_use)
}

fn print_disambiguation_help<'tcx>(
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
    fn_has_self_parameter: bool,
) {
    let mut applicability = Applicability::MachineApplicable;
    let (span, sugg) = if let (ty::AssocKind::Fn, Some(args)) = (kind, args) {
        let args = format!(
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

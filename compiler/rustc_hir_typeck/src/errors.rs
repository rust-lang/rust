// ignore-tidy-file-filelength
// FIXME: we should move the field error reporting code somewhere else.

//! Type checking expressions.
//!
//! See [`rustc_hir_analysis::check`] for more context on type checking in general.

use rustc_abi::FieldIdx;
use rustc_attr_ir::lang_items::LangItem;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, ErrorGuaranteed, MultiSpan, StashKey, Subdiagnostic, listify, pluralize,
    struct_span_code_err,
};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{ExprKind, HirId, is_range_literal};
use rustc_hir_analysis::diagnostics::NoFieldOnType;
use rustc_hir_analysis::hir_ty_lowering::HirTyLowerer as _;
use rustc_middle::ty::print::{with_forced_trimmed_paths, with_no_trimmed_paths};
use rustc_middle::ty::{self, GenericArgs, Ty, TypeVisitableExt, fold_regions};
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::{Ident, Span, Symbol, bug, kw, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::{self, ObligationCauseCode};
use tracing::{debug, instrument};

use crate::diagnostics::{
    HelpUseLatestEdition, ReturnLikeStatementKind, ReturnStmtOutsideOfFnBody, TypeMismatchFruTypo,
};
use crate::{FnCtxt, type_error_struct};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Emit an error because `return` or `become` is used outside of a function body.
    ///
    /// `expr` is the `return` (`become`) "statement", `kind` is the kind of the statement
    /// either `Return` or `Become`.
    pub(crate) fn emit_return_outside_of_fn_body(
        &self,
        expr: &hir::Expr<'_>,
        kind: ReturnLikeStatementKind,
    ) {
        let mut err = ReturnStmtOutsideOfFnBody {
            span: expr.span,
            encl_body_span: None,
            encl_fn_span: None,
            statement_kind: kind,
        };

        let encl_item_id = self.tcx.hir_get_parent_item(expr.hir_id);

        if let hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn { .. },
            span: encl_fn_span,
            ..
        })
        | hir::Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(_)),
            span: encl_fn_span,
            ..
        })
        | hir::Node::ImplItem(hir::ImplItem {
            kind: hir::ImplItemKind::Fn(..),
            span: encl_fn_span,
            ..
        }) = self.tcx.hir_node_by_def_id(encl_item_id.def_id)
        {
            // We are inside a function body, so reporting "return statement
            // outside of function body" needs an explanation.

            let encl_body_owner_id = self.tcx.hir_enclosing_body_owner(expr.hir_id);

            // If this didn't hold, we would not have to report an error in
            // the first place.
            assert_ne!(encl_item_id.def_id, encl_body_owner_id);

            let encl_body = self.tcx.hir_body_owned_by(encl_body_owner_id);

            err.encl_body_span = Some(encl_body.value.span);
            err.encl_fn_span = Some(*encl_fn_span);
        }

        self.dcx().emit_err(err);
    }

    pub(crate) fn point_at_return_for_opaque_ty_error(
        &self,
        errors: &mut ThinVec<traits::FulfillmentError<'tcx>>,
        hir_id: HirId,
        span: Span,
        return_expr_ty: Ty<'tcx>,
        return_span: Span,
    ) {
        // Don't point at the whole block if it's empty
        if span == return_span {
            return;
        }
        for err in errors {
            let cause = &mut err.obligation.cause;
            if let ObligationCauseCode::OpaqueReturnType(None) = cause.code() {
                let new_cause = self.cause(
                    cause.span,
                    ObligationCauseCode::OpaqueReturnType(Some((return_expr_ty, hir_id))),
                );
                *cause = new_cause;
            }
        }
    }

    /// Report an error for a struct field expression when there are fields which aren't provided.
    ///
    /// ```text
    /// error: missing field `you_can_use_this_field` in initializer of `foo::Foo`
    ///  --> src/main.rs:8:5
    ///   |
    /// 8 |     foo::Foo {};
    ///   |     ^^^^^^^^ missing `you_can_use_this_field`
    ///
    /// error: aborting due to 1 previous error
    /// ```
    pub(crate) fn report_missing_fields(
        &self,
        adt_ty: Ty<'tcx>,
        span: Span,
        full_span: Span,
        remaining_fields: UnordMap<Ident, (FieldIdx, &ty::FieldDef)>,
        variant: &'tcx ty::VariantDef,
        hir_fields: &'tcx [hir::ExprField<'tcx>],
        args: GenericArgs<'tcx>,
    ) {
        let len = remaining_fields.len();

        let displayable_field_names: Vec<&str> =
            remaining_fields.items().map(|(ident, _)| ident.as_str()).into_sorted_stable_ord();

        let mut truncated_fields_error = String::new();
        let remaining_fields_names = match &displayable_field_names[..] {
            [field1] => format!("`{field1}`"),
            [field1, field2] => format!("`{field1}` and `{field2}`"),
            [field1, field2, field3] => format!("`{field1}`, `{field2}` and `{field3}`"),
            _ => {
                truncated_fields_error =
                    format!(" and {} other field{}", len - 3, pluralize!(len - 3));
                displayable_field_names
                    .iter()
                    .take(3)
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };

        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0063,
            "missing field{} {}{} in initializer of `{}`",
            pluralize!(len),
            remaining_fields_names,
            truncated_fields_error,
            adt_ty
        );
        err.span_label(span, format!("missing {remaining_fields_names}{truncated_fields_error}"));

        if remaining_fields.items().all(|(_, (_, field))| field.value.is_some())
            && self.tcx.sess.is_nightly_build()
        {
            let msg = format!(
                "all remaining fields have default values, {you_can} use those values with `..`",
                you_can = if self.tcx.features().default_field_values() {
                    "you can"
                } else {
                    "if you added `#![feature(default_field_values)]` to your crate you could"
                },
            );
            if let Some(hir_field) = hir_fields.last() {
                err.span_suggestion_verbose(
                    hir_field.span.shrink_to_hi(),
                    msg,
                    ", ..".to_string(),
                    Applicability::MachineApplicable,
                );
            } else if hir_fields.is_empty() {
                err.span_suggestion_verbose(
                    span.shrink_to_hi().with_hi(full_span.hi()),
                    msg,
                    " { .. }".to_string(),
                    Applicability::MachineApplicable,
                );
            }
        }

        if let Some(hir_field) = hir_fields.last() {
            self.suggest_fru_from_range_and_emit(hir_field, variant, args, err);
        } else {
            err.emit();
        }
    }

    /// If the last field is a range literal, but it isn't supposed to be, then they probably
    /// meant to use functional update syntax.
    pub(crate) fn suggest_fru_from_range_and_emit(
        &self,
        last_expr_field: &hir::ExprField<'tcx>,
        variant: &ty::VariantDef,
        args: GenericArgs<'tcx>,
        mut err: Diag<'_>,
    ) {
        if is_range_literal(last_expr_field.expr)
            && let ExprKind::Struct(&qpath, [range_start, range_end], _) = last_expr_field.expr.kind
            && self.tcx.qpath_is_lang_item(qpath, LangItem::Range)
            && let variant_field =
                variant.fields.iter().find(|field| field.ident(self.tcx) == last_expr_field.ident)
            && let range_def_id = self.tcx.lang_items().range_struct()
            && variant_field
                .and_then(|field| field.ty(self.tcx, args).skip_norm_wip().ty_adt_def())
                .map(|adt| adt.did())
                != range_def_id
        {
            // Use a (somewhat arbitrary) filtering heuristic to avoid printing
            // expressions that are either too long, or have control character
            // such as newlines in them.
            let expr = self
                .tcx
                .sess
                .source_map()
                .span_to_snippet(range_end.expr.span)
                .ok()
                .filter(|s| s.len() < 25 && !s.contains(|c: char| c.is_control()));

            let fru_span = self
                .tcx
                .sess
                .source_map()
                .span_extend_while_whitespace(range_start.expr.span)
                .shrink_to_hi()
                .to(range_end.expr.span);

            err.subdiagnostic(TypeMismatchFruTypo {
                expr_span: range_start.expr.span,
                fru_span,
                expr,
            });

            // Suppress any range expr type mismatches
            self.dcx().try_steal_replace_and_emit_err(
                last_expr_field.span,
                StashKey::MaybeFruTypo,
                err,
            );
        } else {
            err.emit();
        }
    }

    /// Report an error for a struct field expression when there are invisible fields.
    ///
    /// ```text
    /// error: cannot construct `Foo` with struct literal syntax due to private fields
    ///  --> src/main.rs:8:5
    ///   |
    /// 8 |     foo::Foo {};
    ///   |     ^^^^^^^^
    ///
    /// error: aborting due to 1 previous error
    /// ```
    pub(crate) fn report_private_fields(
        &self,
        adt_ty: Ty<'tcx>,
        span: Span,
        expr_span: Span,
        private_fields: Vec<&ty::FieldDef>,
        used_fields: &'tcx [hir::ExprField<'tcx>],
    ) {
        let mut err =
            self.dcx().struct_span_err(
                span,
                format!(
                    "cannot construct `{adt_ty}` with struct literal syntax due to private fields",
                ),
            );
        let (used_private_fields, remaining_private_fields): (
            Vec<(Symbol, Span, bool)>,
            Vec<(Symbol, Span, bool)>,
        ) = private_fields
            .iter()
            .map(|field| {
                match used_fields.iter().find(|used_field| field.name == used_field.ident.name) {
                    Some(used_field) => (field.name, used_field.span, true),
                    None => (field.name, self.tcx.def_span(field.did), false),
                }
            })
            .partition(|field| field.2);
        err.span_labels(used_private_fields.iter().map(|(_, span, _)| *span), "private field");

        if let ty::Adt(def, _) = adt_ty.kind() {
            if (def.did().is_local() || !used_fields.is_empty())
                && !remaining_private_fields.is_empty()
            {
                let names = if remaining_private_fields.len() > 6 {
                    String::new()
                } else {
                    format!(
                        "{} ",
                        listify(&remaining_private_fields, |(name, _, _)| format!("`{name}`"))
                            .expect("expected at least one private field to report")
                    )
                };
                err.note(format!(
                    "{}private field{s} {names}that {were} not provided",
                    if used_fields.is_empty() { "" } else { "...and other " },
                    s = pluralize!(remaining_private_fields.len()),
                    were = pluralize!("was", remaining_private_fields.len()),
                ));
            }

            let def_id = def.did();
            let mut items = self
                .tcx
                .inherent_impls(def_id)
                .into_iter()
                .flat_map(|&i| self.tcx.associated_items(i).in_definition_order())
                // Only assoc fn with no receivers.
                .filter(|item| item.is_fn() && !item.is_method())
                .filter_map(|item| {
                    // Only assoc fns that return `Self`
                    let fn_sig = self
                        .tcx
                        .fn_sig(item.def_id)
                        .instantiate(self.tcx, self.fresh_args_for_item(span, item.def_id))
                        .skip_norm_wip();
                    let ret_ty = self.tcx.instantiate_bound_regions_with_erased(fn_sig.output());
                    if !self.can_eq(self.param_env, ret_ty, adt_ty) {
                        return None;
                    }
                    let input_len = fn_sig.inputs().skip_binder().len();
                    let name = item.name();
                    let order = !name.as_str().starts_with("new");
                    Some((order, name, input_len))
                })
                .collect::<Vec<_>>();
            items.sort_by_key(|(order, _, _)| *order);
            let suggestion = |name, args| {
                format!(
                    "::{name}({})",
                    std::iter::repeat_n("_", args).collect::<Vec<_>>().join(", ")
                )
            };
            match &items[..] {
                [] => {}
                [(_, name, args)] => {
                    err.span_suggestion_verbose(
                        span.shrink_to_hi().with_hi(expr_span.hi()),
                        format!("you might have meant to use the `{name}` associated function"),
                        suggestion(name, *args),
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => {
                    err.span_suggestions(
                        span.shrink_to_hi().with_hi(expr_span.hi()),
                        "you might have meant to use an associated function to build this type",
                        items.iter().map(|(_, name, args)| suggestion(name, *args)),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            if let Some(default_trait) = self.tcx.get_diagnostic_item(sym::Default)
                && self
                    .infcx
                    .type_implements_trait(default_trait, [adt_ty], self.param_env)
                    .may_apply()
            {
                err.multipart_suggestion(
                    "consider using the `Default` trait",
                    vec![
                        (span.shrink_to_lo(), "<".to_string()),
                        (
                            span.shrink_to_hi().with_hi(expr_span.hi()),
                            " as std::default::Default>::default()".to_string(),
                        ),
                    ],
                    Applicability::MaybeIncorrect,
                );
            }
        }

        err.emit();
    }

    pub(crate) fn report_unknown_field(
        &self,
        ty: Ty<'tcx>,
        variant: &'tcx ty::VariantDef,
        expr: &hir::Expr<'_>,
        field: &hir::ExprField<'_>,
        skip_fields: &[hir::ExprField<'_>],
        kind_name: &str,
    ) -> ErrorGuaranteed {
        // we don't care to report errors for a struct if the struct itself is tainted
        if let Err(guar) = variant.has_errors() {
            return guar;
        }
        let mut err = self.err_ctxt().type_error_struct_with_diag(
            field.ident.span,
            |actual| match ty.kind() {
                ty::Adt(adt, ..) if adt.is_enum() => struct_span_code_err!(
                    self.dcx(),
                    field.ident.span,
                    E0559,
                    "{} `{}::{}` has no field named `{}`",
                    kind_name,
                    actual,
                    variant.name,
                    field.ident
                ),
                _ => struct_span_code_err!(
                    self.dcx(),
                    field.ident.span,
                    E0560,
                    "{} `{}` has no field named `{}`",
                    kind_name,
                    actual,
                    field.ident
                ),
            },
            ty,
        );

        let variant_ident_span = self.tcx.def_ident_span(variant.def_id).unwrap();
        match variant.ctor {
            Some((CtorKind::Fn, def_id)) => match ty.kind() {
                ty::Adt(adt, ..) if adt.is_enum() => {
                    err.span_context(self.tcx.def_span(adt.did()).shrink_to_lo());
                    err.span_label(
                        variant_ident_span,
                        format!(
                            "`{adt}::{variant}` defined here",
                            adt = ty,
                            variant = variant.name,
                        ),
                    );
                    err.span_label(field.ident.span, "field does not exist");
                    let fn_sig = self.tcx.fn_sig(def_id).instantiate_identity().skip_norm_wip();
                    let inputs = fn_sig.inputs().skip_binder();
                    let fields = format!(
                        "({})",
                        inputs.iter().map(|i| format!("/* {i} */")).collect::<Vec<_>>().join(", ")
                    );
                    let (replace_span, sugg) = match expr.kind {
                        hir::ExprKind::Struct(qpath, ..) => {
                            (qpath.span().shrink_to_hi().with_hi(expr.span.hi()), fields)
                        }
                        _ => {
                            (expr.span, format!("{ty}::{variant}{fields}", variant = variant.name))
                        }
                    };
                    err.span_suggestion_verbose(
                        replace_span,
                        format!(
                            "`{adt}::{variant}` is a tuple {kind_name}, use the appropriate syntax",
                            adt = ty,
                            variant = variant.name,
                        ),
                        sugg,
                        Applicability::HasPlaceholders,
                    );
                }
                _ => {
                    err.span_label(variant_ident_span, format!("`{ty}` defined here"));
                    err.span_label(field.ident.span, "field does not exist");
                    let fn_sig = self.tcx.fn_sig(def_id).instantiate_identity().skip_norm_wip();
                    let inputs = fn_sig.inputs().skip_binder();
                    let fields = format!(
                        "({})",
                        inputs.iter().map(|i| format!("/* {i} */")).collect::<Vec<_>>().join(", ")
                    );
                    err.span_suggestion_verbose(
                        expr.span,
                        format!("`{ty}` is a tuple {kind_name}, use the appropriate syntax",),
                        format!("{ty}{fields}"),
                        Applicability::HasPlaceholders,
                    );
                }
            },
            _ => {
                // prevent all specified fields from being suggested
                let available_field_names = self.available_field_names(variant, expr, skip_fields);
                if let Some(field_name) =
                    find_best_match_for_name(&available_field_names, field.ident.name, None)
                    && !(field.ident.name.as_str().parse::<usize>().is_ok()
                        && field_name.as_str().parse::<usize>().is_ok())
                {
                    err.span_label(field.ident.span, "unknown field");
                    err.span_suggestion_verbose(
                        field.ident.span,
                        "a field with a similar name exists",
                        field_name,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    match ty.kind() {
                        ty::Adt(adt, ..) => {
                            if adt.is_enum() {
                                err.span_label(
                                    field.ident.span,
                                    format!("`{}::{}` does not have this field", ty, variant.name),
                                );
                            } else {
                                err.span_label(
                                    field.ident.span,
                                    format!("`{ty}` does not have this field"),
                                );
                            }
                            if available_field_names.is_empty() {
                                err.note("all struct fields are already assigned");
                            } else {
                                err.note(format!(
                                    "available fields are: {}",
                                    self.name_series_display(available_field_names)
                                ));
                            }
                        }
                        _ => bug!("non-ADT passed to report_unknown_field"),
                    }
                };
            }
        }
        err.emit_err()
    }

    fn name_series_display(&self, names: Vec<Symbol>) -> String {
        // dynamic limit, to never omit just one field
        let limit = if names.len() == 6 { 6 } else { 5 };
        let mut display =
            names.iter().take(limit).map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ");
        if names.len() > limit {
            display = format!("{} ... and {} others", display, names.len() - limit);
        }
        display
    }

    fn suggest_await_on_field_access(
        &self,
        err: &mut Diag<'_>,
        field_ident: Ident,
        base: &'tcx hir::Expr<'tcx>,
        ty: Ty<'tcx>,
    ) {
        let Some(output_ty) = self.tcx.get_impl_future_output_ty(ty) else {
            err.span_label(field_ident.span, "unknown field");
            return;
        };
        let ty::Adt(def, _) = output_ty.kind() else {
            err.span_label(field_ident.span, "unknown field");
            return;
        };
        // no field access on enum type
        if def.is_enum() {
            err.span_label(field_ident.span, "unknown field");
            return;
        }
        if !def.non_enum_variant().fields.iter().any(|field| field.ident(self.tcx) == field_ident) {
            err.span_label(field_ident.span, "unknown field");
            return;
        }
        err.span_label(
            field_ident.span,
            "field not available in `impl Future`, but it is available in its `Output`",
        );
        match self.tcx.coroutine_kind(self.body_def_id) {
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)) => {
                err.span_suggestion_verbose(
                    base.span.shrink_to_hi(),
                    "consider `await`ing on the `Future` to access the field",
                    ".await",
                    Applicability::MaybeIncorrect,
                );
            }
            _ => {
                let mut span: MultiSpan = base.span.into();
                span.push_span_label(self.tcx.def_span(self.body_def_id), "this is not `async`");
                err.span_note(
                    span,
                    "this implements `Future` and its output type has the field, \
                    but the future cannot be awaited in a synchronous function",
                );
            }
        }
    }

    pub(crate) fn ban_nonexisting_field(
        &self,
        ident: Ident,
        base: &'tcx hir::Expr<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        base_ty: Ty<'tcx>,
    ) -> ErrorGuaranteed {
        debug!(
            "ban_nonexisting_field: field={:?}, base={:?}, expr={:?}, base_ty={:?}",
            ident, base, expr, base_ty
        );
        let mut err = self.no_such_field_err(ident, base_ty, expr);

        match *base_ty.peel_refs().kind() {
            ty::Array(_, len) => {
                self.maybe_suggest_array_indexing(&mut err, base, ident, len);
            }
            ty::RawPtr(..) => {
                self.suggest_first_deref_field(&mut err, base, ident);
            }
            ty::Param(param_ty) => {
                err.span_label(ident.span, "unknown field");
                self.point_at_param_definition(&mut err, param_ty);
            }
            ty::Alias(_, ty::AliasTy { kind: ty::Opaque { .. }, .. }) => {
                self.suggest_await_on_field_access(&mut err, ident, base, base_ty.peel_refs());
            }
            _ => {
                err.span_label(ident.span, "unknown field");
            }
        }

        self.suggest_fn_call(&mut err, base, base_ty, |output_ty| {
            if let ty::Adt(def, _) = output_ty.kind()
                && !def.is_enum()
            {
                def.non_enum_variant().fields.iter().any(|field| {
                    field.ident(self.tcx) == ident
                        && field.vis.is_accessible_from(expr.hir_id.owner.def_id, self.tcx)
                })
            } else if let ty::Tuple(tys) = output_ty.kind()
                && let Ok(idx) = ident.as_str().parse::<usize>()
            {
                idx < tys.len()
            } else {
                false
            }
        });

        if ident.name == kw::Await {
            // We know by construction that `<expr>.await` is either on Rust 2015
            // or results in `ExprKind::Await`. Suggest switching the edition to 2018.
            err.note("to `.await` a `Future`, switch to Rust 2018 or later");
            HelpUseLatestEdition::new().add_to_diag(&mut err);
        }

        err.emit_err()
    }

    pub(crate) fn ban_private_field_access(
        &self,
        expr: &hir::Expr<'tcx>,
        expr_t: Ty<'tcx>,
        field: Ident,
        base_did: DefId,
        return_ty: Option<Ty<'tcx>>,
    ) -> ErrorGuaranteed {
        let mut err = self.private_field_err(field, base_did);

        // Also check if an accessible method exists, which is often what is meant.
        if self.method_exists_for_diagnostic(field, expr_t, expr.hir_id, return_ty)
            && !self.expr_in_place(expr.hir_id)
        {
            self.suggest_method_call(
                &mut err,
                format!("a method `{field}` also exists, call it with parentheses"),
                field,
                expr_t,
                expr,
                None,
            );
        }
        err.emit_err()
    }

    pub(crate) fn ban_take_value_of_method(
        &self,
        expr: &hir::Expr<'tcx>,
        expr_t: Ty<'tcx>,
        field: Ident,
    ) -> ErrorGuaranteed {
        let mut err = type_error_struct!(
            self.dcx(),
            field.span,
            expr_t,
            E0615,
            "attempted to take value of method `{field}` on type `{expr_t}`",
        );
        err.span_label(field.span, "method, not a field");
        let expr_is_call =
            if let hir::Node::Expr(hir::Expr { kind: ExprKind::Call(callee, _args), .. }) =
                self.tcx.parent_hir_node(expr.hir_id)
            {
                expr.hir_id == callee.hir_id
            } else {
                false
            };
        let expr_snippet =
            self.tcx.sess.source_map().span_to_snippet(expr.span).unwrap_or_default();
        let is_wrapped = expr_snippet.starts_with('(') && expr_snippet.ends_with(')');
        let after_open = expr.span.lo() + rustc_span::BytePos(1);
        let before_close = expr.span.hi() - rustc_span::BytePos(1);

        if expr_is_call && is_wrapped {
            err.multipart_suggestion(
                "remove wrapping parentheses to call the method",
                vec![
                    (expr.span.with_hi(after_open), String::new()),
                    (expr.span.with_lo(before_close), String::new()),
                ],
                Applicability::MachineApplicable,
            );
        } else if !self.expr_in_place(expr.hir_id) {
            // Suggest call parentheses inside the wrapping parentheses
            let span = if is_wrapped {
                expr.span.with_lo(after_open).with_hi(before_close)
            } else {
                expr.span
            };
            self.suggest_method_call(
                &mut err,
                "use parentheses to call the method",
                field,
                expr_t,
                expr,
                Some(span),
            );
        } else if let ty::RawPtr(ptr_ty, _) = expr_t.kind()
            && let ty::Adt(adt_def, _) = ptr_ty.kind()
            && let ExprKind::Field(base_expr, _) = expr.kind
            && let [variant] = &adt_def.variants().raw
            && variant.fields.iter().any(|f| f.ident(self.tcx) == field)
        {
            err.multipart_suggestion(
                "to access the field, dereference first",
                vec![
                    (base_expr.span.shrink_to_lo(), "(*".to_string()),
                    (base_expr.span.shrink_to_hi(), ")".to_string()),
                ],
                Applicability::MaybeIncorrect,
            );
        } else {
            err.help("methods are immutable and cannot be assigned to");
        }

        // See `StashKey::GenericInFieldExpr` for more info
        self.dcx().try_steal_replace_and_emit_err(field.span, StashKey::GenericInFieldExpr, err)
    }

    fn point_at_param_definition(&self, err: &mut Diag<'_>, param: ty::ParamTy) {
        let generics = self.tcx.generics_of(self.body_def_id);
        let generic_param = generics.type_param(param, self.tcx);
        if let ty::GenericParamDefKind::Type { synthetic: true, .. } = generic_param.kind {
            return;
        }
        let param_def_id = generic_param.def_id;
        let param_hir_id = match param_def_id.as_local() {
            Some(x) => self.tcx.local_def_id_to_hir_id(x),
            None => return,
        };
        let param_span = self.tcx.hir_span(param_hir_id);
        let param_name = self.tcx.hir_ty_param_name(param_def_id.expect_local());

        err.span_label(param_span, format!("type parameter '{param_name}' declared here"));
    }

    fn maybe_suggest_array_indexing(
        &self,
        err: &mut Diag<'_>,
        base: &hir::Expr<'_>,
        field: Ident,
        len: ty::Const<'tcx>,
    ) {
        err.span_label(field.span, "unknown field");
        if let (Some(len), Ok(user_index)) = (
            self.try_structurally_resolve_const(base.span, len).try_to_target_usize(self.tcx),
            field.as_str().parse::<u64>(),
        ) {
            let help = "instead of using tuple indexing, use array indexing";
            let applicability = if len < user_index {
                Applicability::MachineApplicable
            } else {
                Applicability::MaybeIncorrect
            };
            err.multipart_suggestion(
                help,
                vec![
                    (base.span.between(field.span), "[".to_string()),
                    (field.span.shrink_to_hi(), "]".to_string()),
                ],
                applicability,
            );
        }
    }

    fn suggest_first_deref_field(&self, err: &mut Diag<'_>, base: &hir::Expr<'_>, field: Ident) {
        err.span_label(field.span, "unknown field");
        if base.span.from_expansion() || field.span.from_expansion() {
            return;
        }
        let val = if let Ok(base) = self.tcx.sess.source_map().span_to_snippet(base.span)
            && base.len() < 20
        {
            format!("`{base}`")
        } else {
            "the value".to_string()
        };
        err.multipart_suggestion(
            format!("{val} is a raw pointer; try dereferencing it"),
            vec![
                (base.span.shrink_to_lo(), "(*".into()),
                (base.span.between(field.span), format!(").")),
            ],
            Applicability::MaybeIncorrect,
        );
    }

    pub(crate) fn no_such_field_err(
        &self,
        field: Ident,
        base_ty: Ty<'tcx>,
        expr: &hir::Expr<'tcx>,
    ) -> Diag<'_> {
        let span = field.span;
        debug!("no_such_field_err(span: {:?}, field: {:?}, expr_t: {:?})", span, field, base_ty);

        let mut err = self.dcx().create_err(NoFieldOnType { span, ty: base_ty, field });
        if base_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        self.detect_shadowed_method_on_incorrect_field_access(&mut err, field, base_ty, expr);

        if let Some(within_macro_span) = span.within_macro(expr.span, self.tcx.sess.source_map()) {
            err.span_label(within_macro_span, "due to this macro variable");
        }

        // Check if there is an associated function with the same name.
        if let Some(def_id) = base_ty.peel_refs().ty_adt_def().map(|d| d.did()) {
            for &impl_def_id in self.tcx.inherent_impls(def_id) {
                for item in self.tcx.associated_items(impl_def_id).in_definition_order() {
                    if let ExprKind::Field(base_expr, _) = expr.kind
                        && item.name() == field.name
                        && matches!(item.kind, ty::AssocKind::Fn { has_self: false, .. })
                    {
                        err.span_label(field.span, "this is an associated function, not a method");
                        err.note("found the following associated function; to be used as method, it must have a `self` parameter");
                        let impl_ty =
                            self.tcx.type_of(impl_def_id).instantiate_identity().skip_norm_wip();
                        err.span_note(
                            self.tcx.def_span(item.def_id),
                            format!("the candidate is defined in an impl for the type `{impl_ty}`"),
                        );

                        let ty_str = match base_ty.peel_refs().kind() {
                            ty::Adt(def, args) => self.tcx.def_path_str_with_args(def.did(), *args),
                            _ => base_ty.peel_refs().to_string(),
                        };
                        err.multipart_suggestion(
                            "use associated function syntax instead",
                            vec![
                                (base_expr.span, ty_str),
                                (base_expr.span.between(field.span), "::".to_string()),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                        return err;
                    }
                }
            }
        }

        // try to add a suggestion in case the field is a nested field of a field of the Adt
        let (ty, unwrap) = if let ty::Adt(def, args) = base_ty.kind()
            && (self.tcx.is_diagnostic_item(sym::Result, def.did())
                || self.tcx.is_diagnostic_item(sym::Option, def.did()))
            && let Some(arg) = args.get(0)
            && let Some(ty) = arg.as_type()
        {
            (ty, "unwrap().")
        } else {
            (base_ty, "")
        };
        for found_fields in self.get_field_candidates_considering_privacy_for_diag(span, ty) {
            let field_names = found_fields.iter().map(|field| field.0.name).collect::<Vec<_>>();
            let mut candidate_fields: Vec<_> = found_fields
                .into_iter()
                .filter_map(|candidate_field| {
                    self.check_for_nested_field_satisfying_condition_for_diag(
                        span,
                        &|candidate_field, _| candidate_field == field,
                        candidate_field,
                        vec![],
                    )
                })
                .map(|mut field_path| {
                    field_path.pop();
                    field_path.iter().map(|id| format!("{}.", id)).collect::<String>()
                })
                .collect::<Vec<_>>();
            candidate_fields.sort();

            let len = candidate_fields.len();
            // Don't suggest `.field` if the base expr is from a different
            // syntax context than the field.
            if len > 0 && expr.span.ctxt() == field.span.ctxt() {
                err.span_suggestions(
                    field.span.shrink_to_lo(),
                    format!(
                        "{} of the expressions' fields {} a field of the same name",
                        if len > 1 { "some" } else { "one" },
                        if len > 1 { "have" } else { "has" },
                    ),
                    candidate_fields.iter().map(|path| format!("{unwrap}{path}")),
                    Applicability::MaybeIncorrect,
                );
            } else if let Some(field_name) =
                find_best_match_for_name(&field_names, field.name, None)
                && !(field.name.as_str().parse::<usize>().is_ok()
                    && field_name.as_str().parse::<usize>().is_ok())
            {
                err.span_suggestion_verbose(
                    field.span,
                    "a field with a similar name exists",
                    format!("{unwrap}{}", field_name),
                    Applicability::MaybeIncorrect,
                );
            } else if !field_names.is_empty() {
                let is = if field_names.len() == 1 { " is" } else { "s are" };
                err.note(
                    format!("available field{is}: {}", self.name_series_display(field_names),),
                );
            }
        }
        err
    }

    fn detect_shadowed_method_on_incorrect_field_access(
        &self,
        err: &mut Diag<'_>,
        field: Ident,
        base_ty: Ty<'tcx>,
        expr: &hir::Expr<'tcx>,
    ) {
        let hir::ExprKind::Field(expr, ..) = expr.kind else { return };
        self.find_shadowed_method_and_provide_diagnostic_context(err, base_ty, expr, |ty| {
            // Does the alternative method call's resulting type auto-deref to a type that has
            // the field the user is trying to access?
            self.get_field_candidates_considering_privacy_for_diag(expr.span, ty)
                .into_iter()
                .flat_map(|fields| fields.into_iter())
                .find(|(f, _ty)| f.name == field.name)
                .is_some()
        });
    }

    pub(crate) fn all_methods(
        &self,
        method_ident: Ident,
        ty: Ty<'tcx>,
    ) -> impl Iterator<Item = (DefId, DefId, bool)> {
        let mut impls: Vec<_> = self
            .tcx
            .all_traits_including_private()
            .filter(|trait_def_id| {
                // Consider only traits with the associated type
                self.tcx
                    .associated_items(*trait_def_id)
                    .in_definition_order()
                    .any(|i| i.is_method() && i.ident(self.tcx).name == method_ident.name)
                    && self.tcx.all_impls(*trait_def_id).any(|impl_def_id| {
                        let header = self.tcx.impl_trait_header(impl_def_id);
                        let trait_ref = header
                            .trait_ref
                            .instantiate(
                                self.tcx,
                                self.infcx.fresh_args_for_item(method_ident.span, impl_def_id),
                            )
                            .skip_norm_wip();
                        let value = fold_regions(self.tcx, ty, |_, _| self.tcx.lifetimes.re_erased);
                        if value.has_escaping_bound_vars() {
                            return false;
                        }
                        self.infcx.can_eq(ty::ParamEnv::empty(), trait_ref.self_ty(), value)
                            && header.polarity != ty::ImplPolarity::Negative
                    })
            })
            .map(|def_id| (def_id, false))
            .collect();
        if let ty::Adt(def, _args) = ty.kind() {
            // Get all the `impl ReceiverType`s.
            impls.extend(
                self.tcx
                    .inherent_impls(def.did())
                    .into_iter()
                    .cloned()
                    .map(|def_id| (def_id, true)),
            );
        }
        impls.into_iter().filter_map(move |(i, inherent)| {
            self.tcx
                .associated_items(i)
                .in_definition_order()
                .find(|assoc| assoc.is_method() && assoc.ident(self.tcx).name == method_ident.name)
                .map(|assoc| (i, assoc.def_id, inherent))
        })
    }

    /// Given `expr`, detect if it is a method call that might have been shadowed by another trait
    /// being in scope. Only include those traits/impls whose similarly named method have a return
    /// type that `finder` considers as "valid" (for example, it has the expected field or method).
    pub(crate) fn find_shadowed_method_and_provide_diagnostic_context(
        &self,
        err: &mut Diag<'_>,
        base_ty: Ty<'tcx>,
        mut expr: &hir::Expr<'tcx>,
        finder: impl Fn(Ty<'tcx>) -> bool,
    ) {
        if let hir::ExprKind::Unary(hir::UnOp::Deref, e) = expr.kind {
            // If we have `*foo`, we want `foo`, which will subsequently be resolved to the earlier
            // binding.
            expr = e;
        }
        if let hir::ExprKind::Path(hir::QPath::Resolved(
            None,
            hir::Path { res: Res::Local(hir_id), .. },
        )) = expr.kind
            && let hir::Node::Pat(_) = self.tcx.hir_node(*hir_id)
            && let hir::Node::LetStmt(local) = self.tcx.parent_hir_node(*hir_id)
            && let Some(binding_expr) = local.init
        {
            // Get and point at the binding's prior method call, if it is one:
            // ```
            // let foo = bar.borrow();
            // foo.qux();
            // ```
            // Where `qux` is present in the return type of `<Bar as RefCell>::borrow`, but not
            // `<Bar as Borrow>::borrow`.
            expr = binding_expr;
        };
        let hir::ExprKind::MethodCall(segment, rcvr, args, span_with_args) = expr.kind else {
            return;
        };
        let Some(def_id) = self.typeck_results.borrow().type_dependent_def_id(expr.hir_id) else {
            return;
        };
        let rcvr_ty = self.typeck_results.borrow().expr_ty_adjusted(rcvr);
        let autoderef = self.autoderef(span_with_args, rcvr_ty).silence_errors();
        let mut alternatives = FxIndexSet::default();
        for (ty, _) in autoderef {
            // Get all traits that are implemented by the potential autoderefs for the
            // receiver for methods with the same name.
            let impls = self.all_methods(segment.ident, ty);
            // In all the `impl`s for the autoderefs of the receiver type, look for the method
            // *except* the one that was actually called. This would end up being an empty set
            // if there are no alternatives.
            for (parent, def_id, inherent) in impls.filter(|(_, did, _)| def_id != *did) {
                let gen_args = if let ty::Adt(_, args) = ty.peel_refs().kind()
                    && inherent
                {
                    // We want the arguments of the *type* to be passed to the `impl ReceiverType`.
                    *args
                } else {
                    // We want to use the `ReceiverType` for `Self` on trait impls.
                    GenericArgs::for_item(self.tcx, def_id, |param, _| {
                        if let ty::GenericParamDefKind::Lifetime = param.kind {
                            self.tcx.lifetimes.re_erased.into()
                        } else if param.name == kw::SelfUpper {
                            ty.into()
                        } else {
                            self.infcx.var_for_def(span_with_args, param)
                        }
                    })
                };
                let fn_sig =
                    self.tcx.fn_sig(def_id).instantiate(self.tcx, gen_args).skip_norm_wip();
                let ret_ty = fn_sig.output();
                let ret_ty = self.tcx.instantiate_bound_regions_with_erased(
                    self.deeply_resolve_ignoring_regions_with_obligations(ret_ty),
                );

                // Does the alternative method call's resulting type auto-deref to a type that has
                // the thing the user is trying to access?
                if !finder(ret_ty) {
                    continue;
                }
                // It does! We'll point at it and suggest the fully-qualified path to the
                // alternative method call.
                let mut span: MultiSpan = self.tcx.def_span(def_id).into();
                span.push_span_context(self.tcx.def_span(parent));
                let path = with_no_trimmed_paths!(self.tcx.def_path_str(def_id));
                let after_rcvr = match args {
                    [] => (rcvr.span.shrink_to_hi().to(span_with_args), ")".to_string()),
                    [first, ..] => (rcvr.span.shrink_to_hi().to(first.span), ", ".to_string()),
                };
                let self_ty = fn_sig.input(0).skip_binder();
                let mutbl = match self_ty.kind() {
                    ty::Ref(_, _, mutbl) => mutbl.ref_prefix_str(),
                    _ => "",
                };
                // FIXME: in the suggestions we are now (almost) always suggesting the type or trait
                // name as if it was already imported, regardless of whether it is. This means that
                // a follow up suggestion will be needed if the item is not in scope (which likely
                // won't be).
                let call = if inherent {
                    let args = if let ty::Adt(def, args) = ty.peel_refs().kind() {
                        self.fresh_args_for_item(expr.span, parent).rebase_onto(
                            self.tcx,
                            def.did(),
                            *args,
                        )
                    } else {
                        self.fresh_args_for_item(expr.span, def_id)
                    };
                    // `Type::method(&rcvr)`
                    with_forced_trimmed_paths!(format!(
                        "{}({mutbl}",
                        self.tcx.value_path_str_with_args(def_id, args),
                    ))
                } else {
                    // `<_ as Trait>::method(&rcvr)`
                    format!(
                        "<_ as {}>::{}({mutbl}",
                        with_forced_trimmed_paths!(self.tcx.def_path_str_with_args(
                            parent,
                            self.fresh_args_for_item(expr.span, parent)
                        )),
                        segment.ident
                    )
                };
                let sugg = vec![(rcvr.span.shrink_to_lo(), call), after_rcvr];
                alternatives.insert((path, span, sugg));
            }
        }
        if alternatives.is_empty() {
            return;
        }
        let path = with_no_trimmed_paths!(self.tcx.def_path_str(def_id));

        // On the main diagnostic, point at the method call before the field access to make it clear
        // what it actually was.
        err.span_label(segment.ident.span, format!("this method is `{path}`"));

        // Point at the method definition, including its impl or trait, being explicit about what
        // type it returns, which at this point we know doesn't have the desired field.
        let base_ty = self.tcx.short_string(base_ty, err.long_ty_path());
        let mut span: MultiSpan = self.tcx.def_span(def_id).into();
        span.push_span_context(self.tcx.def_span(self.tcx.parent(def_id)));
        err.span_note(span, format!("this returns `{base_ty}`"));

        for (path, span, sugg) in alternatives {
            // Show the alternatives' method definition...
            err.span_note(span, format!("you might have meant to call `{path}` instead"));
            // ...and suggest the fully-qualified path to call them instead.
            err.multipart_suggestion(
                format!("consider calling `{path}` using the fully-qualified path"),
                sugg,
                Applicability::MaybeIncorrect,
            );
        }
    }

    pub(crate) fn private_field_err(&self, field: Ident, base_did: DefId) -> Diag<'_> {
        let struct_path = self.tcx().def_path_str(base_did);
        let kind_name = self.tcx().def_descr(base_did);
        struct_span_code_err!(
            self.dcx(),
            field.span,
            E0616,
            "field `{field}` of {kind_name} `{struct_path}` is private",
        )
        .with_span_label(field.span, "private field")
    }

    pub(crate) fn get_field_candidates_considering_privacy_for_diag(
        &self,
        span: Span,
        base_ty: Ty<'tcx>,
    ) -> Vec<Vec<(Ident, Ty<'tcx>)>> {
        debug!("get_field_candidates(span: {:?}, base_t: {:?}", span, base_ty);

        let mut autoderef = self.autoderef(span, base_ty).silence_errors();
        let deref_chain: Vec<_> = autoderef.by_ref().collect();

        // Don't probe if we hit the recursion limit, since it may result in
        // quadratic blowup if we then try to further deref the results of this
        // function. This is a best-effort method, after all.
        if autoderef.reached_recursion_limit() {
            return vec![];
        }

        deref_chain
            .into_iter()
            .filter_map(move |(base_t, _)| {
                match base_t.kind() {
                    ty::Adt(base_def, args) if !base_def.is_enum() => {
                        let tcx = self.tcx;
                        let fields = &base_def.non_enum_variant().fields;
                        // Some struct, e.g. some that impl `Deref`, have all private fields
                        // because you're expected to deref them to access the _real_ fields.
                        // This, for example, will help us suggest accessing a field through a `Box<T>`.
                        if fields
                            .iter()
                            .all(|field| !field.vis.is_accessible_from(self.mod_id, tcx))
                        {
                            return None;
                        }
                        return Some(
                            fields
                                .iter()
                                .filter(move |field| {
                                    field.vis.is_accessible_from(self.mod_id, tcx)
                                        && self.is_field_suggestable(field, span)
                                })
                                // For compile-time reasons put a limit on number of fields we search
                                .take(100)
                                .map(|field_def| {
                                    (
                                        field_def.ident(self.tcx).normalize_to_macros_2_0(),
                                        field_def.ty(self.tcx, *args).skip_norm_wip(),
                                    )
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    ty::Tuple(types) => {
                        return Some(
                            types
                                .iter()
                                .enumerate()
                                // For compile-time reasons put a limit on number of fields we search
                                .take(100)
                                .map(|(i, ty)| (Ident::from_str(&i.to_string()), ty))
                                .collect::<Vec<_>>(),
                        );
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// This method is called after we have encountered a missing field error to recursively
    /// search for the field
    #[instrument(skip(self, matches), level = "debug")]
    pub(crate) fn check_for_nested_field_satisfying_condition_for_diag(
        &self,
        span: Span,
        matches: &impl Fn(Ident, Ty<'tcx>) -> bool,
        (candidate_name, candidate_ty): (Ident, Ty<'tcx>),
        mut field_path: Vec<Ident>,
    ) -> Option<Vec<Ident>> {
        if field_path.len() > 3 {
            // For compile-time reasons and to avoid infinite recursion we only check for fields
            // up to a depth of three
            return None;
        }
        field_path.push(candidate_name);
        if matches(candidate_name, candidate_ty) {
            return Some(field_path);
        }
        for nested_fields in
            self.get_field_candidates_considering_privacy_for_diag(span, candidate_ty)
        {
            // recursively search fields of `candidate_field` if it's a ty::Adt
            for field in nested_fields {
                if let Some(field_path) = self.check_for_nested_field_satisfying_condition_for_diag(
                    span,
                    matches,
                    field,
                    field_path.clone(),
                ) {
                    return Some(field_path);
                }
            }
        }
        None
    }

    pub(crate) fn point_at_index(
        &self,
        errors: &mut ThinVec<traits::FulfillmentError<'tcx>>,
        span: Span,
    ) {
        let mut seen_preds = FxHashSet::default();
        // We re-sort here so that the outer most root obligations comes first, as we have the
        // subsequent weird logic to identify *every* relevant obligation for proper deduplication
        // of diagnostics.
        errors.sort_by_key(|error| error.root_obligation.recursion_depth);
        for error in errors {
            match (
                error.root_obligation.predicate.kind().skip_binder(),
                error.obligation.predicate.kind().skip_binder(),
            ) {
                (ty::PredicateKind::Clause(ty::ClauseKind::Trait(predicate)), _)
                    if self.tcx.is_lang_item(predicate.trait_ref.def_id, LangItem::Index) =>
                {
                    seen_preds.insert(error.obligation.predicate.kind().skip_binder());
                }
                (_, ty::PredicateKind::Clause(ty::ClauseKind::Trait(predicate)))
                    if self.tcx.is_diagnostic_item(sym::SliceIndex, predicate.trait_ref.def_id) =>
                {
                    seen_preds.insert(error.obligation.predicate.kind().skip_binder());
                }
                (root, pred) if seen_preds.contains(&pred) || seen_preds.contains(&root) => {}
                _ => continue,
            }
            error.obligation.cause.span = span;
        }
    }
}

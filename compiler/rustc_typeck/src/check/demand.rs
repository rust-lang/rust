use crate::check::FnCtxt;
use rustc_infer::infer::InferOk;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::ObligationCause;

use rustc_ast::util::parser::PREC_POSTFIX;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{is_range_literal, Node};
use rustc_middle::ty::adjustment::AllowTwoPhase;
use rustc_middle::ty::{self, AssocItem, Ty, TypeAndMut};
use rustc_span::symbol::sym;
use rustc_span::Span;

use super::method::probe;

use std::fmt;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn emit_coerce_suggestions(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) {
        self.annotate_expected_due_to_let_ty(err, expr);
        self.suggest_compatible_variants(err, expr, expected, expr_ty);
        self.suggest_deref_ref_or_into(err, expr, expected, expr_ty, expected_ty_expr);
        if self.suggest_calling_boxed_future_when_appropriate(err, expr, expected, expr_ty) {
            return;
        }
        self.suggest_no_capture_closure(err, expected, expr_ty);
        self.suggest_boxing_when_appropriate(err, expr, expected, expr_ty);
        self.suggest_missing_parentheses(err, expr);
        self.note_need_for_fn_pointer(err, expected, expr_ty);
        self.note_internal_mutation_in_method(err, expr, expected, expr_ty);
    }

    // Requires that the two types unify, and prints an error message if
    // they don't.
    pub fn demand_suptype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        if let Some(mut e) = self.demand_suptype_diag(sp, expected, actual) {
            e.emit();
        }
    }

    pub fn demand_suptype_diag(
        &self,
        sp: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Option<DiagnosticBuilder<'tcx>> {
        self.demand_suptype_with_origin(&self.misc(sp), expected, actual)
    }

    pub fn demand_suptype_with_origin(
        &self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Option<DiagnosticBuilder<'tcx>> {
        match self.at(cause, self.param_env).sup(expected, actual) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
                None
            }
            Err(e) => Some(self.report_mismatched_types(&cause, expected, actual, e)),
        }
    }

    pub fn demand_eqtype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        if let Some(mut err) = self.demand_eqtype_diag(sp, expected, actual) {
            err.emit();
        }
    }

    pub fn demand_eqtype_diag(
        &self,
        sp: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Option<DiagnosticBuilder<'tcx>> {
        self.demand_eqtype_with_origin(&self.misc(sp), expected, actual)
    }

    pub fn demand_eqtype_with_origin(
        &self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
    ) -> Option<DiagnosticBuilder<'tcx>> {
        match self.at(cause, self.param_env).eq(expected, actual) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
                None
            }
            Err(e) => Some(self.report_mismatched_types(cause, expected, actual, e)),
        }
    }

    pub fn demand_coerce(
        &self,
        expr: &hir::Expr<'_>,
        checked_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        allow_two_phase: AllowTwoPhase,
    ) -> Ty<'tcx> {
        let (ty, err) =
            self.demand_coerce_diag(expr, checked_ty, expected, expected_ty_expr, allow_two_phase);
        if let Some(mut err) = err {
            err.emit();
        }
        ty
    }

    /// Checks that the type of `expr` can be coerced to `expected`.
    ///
    /// N.B., this code relies on `self.diverges` to be accurate. In particular, assignments to `!`
    /// will be permitted if the diverges flag is currently "always".
    pub fn demand_coerce_diag(
        &self,
        expr: &hir::Expr<'_>,
        checked_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
        allow_two_phase: AllowTwoPhase,
    ) -> (Ty<'tcx>, Option<DiagnosticBuilder<'tcx>>) {
        let expected = self.resolve_vars_with_obligations(expected);

        let e = match self.try_coerce(expr, checked_ty, expected, allow_two_phase) {
            Ok(ty) => return (ty, None),
            Err(e) => e,
        };

        let expr = expr.peel_drop_temps();
        let cause = self.misc(expr.span);
        let expr_ty = self.resolve_vars_with_obligations(checked_ty);
        let mut err = self.report_mismatched_types(&cause, expected, expr_ty, e);

        if self.is_assign_to_bool(expr, expected) {
            // Error reported in `check_assign` so avoid emitting error again.
            err.delay_as_bug();
            return (expected, None);
        }

        self.emit_coerce_suggestions(&mut err, expr, expr_ty, expected, expected_ty_expr);

        (expected, Some(err))
    }

    fn annotate_expected_due_to_let_ty(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
    ) {
        let parent = self.tcx.hir().get_parent_node(expr.hir_id);
        if let Some(hir::Node::Local(hir::Local { ty: Some(ty), init: Some(init), .. })) =
            self.tcx.hir().find(parent)
        {
            if init.hir_id == expr.hir_id {
                // Point at `let` assignment type.
                err.span_label(ty.span, "expected due to this");
            }
        }
    }

    /// Returns whether the expected type is `bool` and the expression is `x = y`.
    pub fn is_assign_to_bool(&self, expr: &hir::Expr<'_>, expected: Ty<'tcx>) -> bool {
        if let hir::ExprKind::Assign(..) = expr.kind {
            return expected == self.tcx.types.bool;
        }
        false
    }

    /// If the expected type is an enum (Issue #55250) with any variants whose
    /// sole field is of the found type, suggest such variants. (Issue #42764)
    fn suggest_compatible_variants(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        expr_ty: Ty<'tcx>,
    ) {
        if let ty::Adt(expected_adt, substs) = expected.kind() {
            if !expected_adt.is_enum() {
                return;
            }

            let mut compatible_variants = expected_adt
                .variants
                .iter()
                .filter(|variant| variant.fields.len() == 1)
                .filter_map(|variant| {
                    let sole_field = &variant.fields[0];
                    let sole_field_ty = sole_field.ty(self.tcx, substs);
                    if self.can_coerce(expr_ty, sole_field_ty) {
                        let variant_path = self.tcx.def_path_str(variant.def_id);
                        // FIXME #56861: DRYer prelude filtering
                        Some(variant_path.trim_start_matches("std::prelude::v1::").to_string())
                    } else {
                        None
                    }
                })
                .peekable();

            if compatible_variants.peek().is_some() {
                if let Ok(expr_text) = self.tcx.sess.source_map().span_to_snippet(expr.span) {
                    let suggestions = compatible_variants.map(|v| format!("{}({})", v, expr_text));
                    let msg = "try using a variant of the expected enum";
                    err.span_suggestions(
                        expr.span,
                        msg,
                        suggestions,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    pub fn get_conversion_methods(
        &self,
        span: Span,
        expected: Ty<'tcx>,
        checked_ty: Ty<'tcx>,
        hir_id: hir::HirId,
    ) -> Vec<AssocItem> {
        let mut methods =
            self.probe_for_return_type(span, probe::Mode::MethodCall, expected, checked_ty, hir_id);
        methods.retain(|m| {
            self.has_only_self_parameter(m)
                && self
                    .tcx
                    .get_attrs(m.def_id)
                    .iter()
                    // This special internal attribute is used to permit
                    // "identity-like" conversion methods to be suggested here.
                    //
                    // FIXME (#46459 and #46460): ideally
                    // `std::convert::Into::into` and `std::borrow:ToOwned` would
                    // also be `#[rustc_conversion_suggestion]`, if not for
                    // method-probing false-positives and -negatives (respectively).
                    //
                    // FIXME? Other potential candidate methods: `as_ref` and
                    // `as_mut`?
                    .any(|a| self.sess().check_name(a, sym::rustc_conversion_suggestion))
        });

        methods
    }

    /// This function checks whether the method is not static and does not accept other parameters than `self`.
    fn has_only_self_parameter(&self, method: &AssocItem) -> bool {
        match method.kind {
            ty::AssocKind::Fn => {
                method.fn_has_self_parameter
                    && self.tcx.fn_sig(method.def_id).inputs().skip_binder().len() == 1
            }
            _ => false,
        }
    }

    /// Identify some cases where `as_ref()` would be appropriate and suggest it.
    ///
    /// Given the following code:
    /// ```
    /// struct Foo;
    /// fn takes_ref(_: &Foo) {}
    /// let ref opt = Some(Foo);
    ///
    /// opt.map(|param| takes_ref(param));
    /// ```
    /// Suggest using `opt.as_ref().map(|param| takes_ref(param));` instead.
    ///
    /// It only checks for `Option` and `Result` and won't work with
    /// ```
    /// opt.map(|param| { takes_ref(param) });
    /// ```
    fn can_use_as_ref(&self, expr: &hir::Expr<'_>) -> Option<(Span, &'static str, String)> {
        let path = match expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(_, ref path)) => path,
            _ => return None,
        };

        let local_id = match path.res {
            hir::def::Res::Local(id) => id,
            _ => return None,
        };

        let local_parent = self.tcx.hir().get_parent_node(local_id);
        let param_hir_id = match self.tcx.hir().find(local_parent) {
            Some(Node::Param(hir::Param { hir_id, .. })) => hir_id,
            _ => return None,
        };

        let param_parent = self.tcx.hir().get_parent_node(*param_hir_id);
        let (expr_hir_id, closure_fn_decl) = match self.tcx.hir().find(param_parent) {
            Some(Node::Expr(hir::Expr {
                hir_id,
                kind: hir::ExprKind::Closure(_, decl, ..),
                ..
            })) => (hir_id, decl),
            _ => return None,
        };

        let expr_parent = self.tcx.hir().get_parent_node(*expr_hir_id);
        let hir = self.tcx.hir().find(expr_parent);
        let closure_params_len = closure_fn_decl.inputs.len();
        let (method_path, method_span, method_expr) = match (hir, closure_params_len) {
            (
                Some(Node::Expr(hir::Expr {
                    kind: hir::ExprKind::MethodCall(path, span, expr, _),
                    ..
                })),
                1,
            ) => (path, span, expr),
            _ => return None,
        };

        let self_ty = self.typeck_results.borrow().node_type(method_expr[0].hir_id);
        let self_ty = format!("{:?}", self_ty);
        let name = method_path.ident.name;
        let is_as_ref_able = (self_ty.starts_with("&std::option::Option")
            || self_ty.starts_with("&std::result::Result")
            || self_ty.starts_with("std::option::Option")
            || self_ty.starts_with("std::result::Result"))
            && (name == sym::map || name == sym::and_then);
        match (is_as_ref_able, self.sess().source_map().span_to_snippet(*method_span)) {
            (true, Ok(src)) => {
                let suggestion = format!("as_ref().{}", src);
                Some((*method_span, "consider using `as_ref` instead", suggestion))
            }
            _ => None,
        }
    }

    crate fn is_hir_id_from_struct_pattern_shorthand_field(
        &self,
        hir_id: hir::HirId,
        sp: Span,
    ) -> bool {
        let sm = self.sess().source_map();
        let parent_id = self.tcx.hir().get_parent_node(hir_id);
        if let Some(parent) = self.tcx.hir().find(parent_id) {
            // Account for fields
            if let Node::Expr(hir::Expr { kind: hir::ExprKind::Struct(_, fields, ..), .. }) = parent
            {
                if let Ok(src) = sm.span_to_snippet(sp) {
                    for field in *fields {
                        if field.ident.as_str() == src && field.is_shorthand {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// This function is used to determine potential "simple" improvements or users' errors and
    /// provide them useful help. For example:
    ///
    /// ```
    /// fn some_fn(s: &str) {}
    ///
    /// let x = "hey!".to_owned();
    /// some_fn(x); // error
    /// ```
    ///
    /// No need to find every potential function which could make a coercion to transform a
    /// `String` into a `&str` since a `&` would do the trick!
    ///
    /// In addition of this check, it also checks between references mutability state. If the
    /// expected is mutable but the provided isn't, maybe we could just say "Hey, try with
    /// `&mut`!".
    pub fn check_ref(
        &self,
        expr: &hir::Expr<'_>,
        checked_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
    ) -> Option<(Span, &'static str, String, Applicability)> {
        let sm = self.sess().source_map();
        let sp = expr.span;
        if sm.is_imported(sp) {
            // Ignore if span is from within a macro #41858, #58298. We previously used the macro
            // call span, but that breaks down when the type error comes from multiple calls down.
            return None;
        }

        let replace_prefix = |s: &str, old: &str, new: &str| {
            s.strip_prefix(old).map(|stripped| new.to_string() + stripped)
        };

        let is_struct_pat_shorthand_field =
            self.is_hir_id_from_struct_pattern_shorthand_field(expr.hir_id, sp);

        // If the span is from a macro, then it's hard to extract the text
        // and make a good suggestion, so don't bother.
        let is_macro = sp.from_expansion() && sp.desugaring_kind().is_none();

        // `ExprKind::DropTemps` is semantically irrelevant for these suggestions.
        let expr = expr.peel_drop_temps();

        match (&expr.kind, expected.kind(), checked_ty.kind()) {
            (_, &ty::Ref(_, exp, _), &ty::Ref(_, check, _)) => match (exp.kind(), check.kind()) {
                (&ty::Str, &ty::Array(arr, _) | &ty::Slice(arr)) if arr == self.tcx.types.u8 => {
                    if let hir::ExprKind::Lit(_) = expr.kind {
                        if let Ok(src) = sm.span_to_snippet(sp) {
                            if let Some(src) = replace_prefix(&src, "b\"", "\"") {
                                return Some((
                                    sp,
                                    "consider removing the leading `b`",
                                    src,
                                    Applicability::MachineApplicable,
                                ));
                            }
                        }
                    }
                }
                (&ty::Array(arr, _) | &ty::Slice(arr), &ty::Str) if arr == self.tcx.types.u8 => {
                    if let hir::ExprKind::Lit(_) = expr.kind {
                        if let Ok(src) = sm.span_to_snippet(sp) {
                            if let Some(src) = replace_prefix(&src, "\"", "b\"") {
                                return Some((
                                    sp,
                                    "consider adding a leading `b`",
                                    src,
                                    Applicability::MachineApplicable,
                                ));
                            }
                        }
                    }
                }
                _ => {}
            },
            (_, &ty::Ref(_, _, mutability), _) => {
                // Check if it can work when put into a ref. For example:
                //
                // ```
                // fn bar(x: &mut i32) {}
                //
                // let x = 0u32;
                // bar(&x); // error, expected &mut
                // ```
                let ref_ty = match mutability {
                    hir::Mutability::Mut => {
                        self.tcx.mk_mut_ref(self.tcx.mk_region(ty::ReStatic), checked_ty)
                    }
                    hir::Mutability::Not => {
                        self.tcx.mk_imm_ref(self.tcx.mk_region(ty::ReStatic), checked_ty)
                    }
                };
                if self.can_coerce(ref_ty, expected) {
                    let mut sugg_sp = sp;
                    if let hir::ExprKind::MethodCall(ref segment, sp, ref args, _) = expr.kind {
                        let clone_trait = self.tcx.require_lang_item(LangItem::Clone, Some(sp));
                        if let ([arg], Some(true), sym::clone) = (
                            &args[..],
                            self.typeck_results.borrow().type_dependent_def_id(expr.hir_id).map(
                                |did| {
                                    let ai = self.tcx.associated_item(did);
                                    ai.container == ty::TraitContainer(clone_trait)
                                },
                            ),
                            segment.ident.name,
                        ) {
                            // If this expression had a clone call when suggesting borrowing
                            // we want to suggest removing it because it'd now be unnecessary.
                            sugg_sp = arg.span;
                        }
                    }
                    if let Ok(src) = sm.span_to_snippet(sugg_sp) {
                        let needs_parens = match expr.kind {
                            // parenthesize if needed (Issue #46756)
                            hir::ExprKind::Cast(_, _) | hir::ExprKind::Binary(_, _, _) => true,
                            // parenthesize borrows of range literals (Issue #54505)
                            _ if is_range_literal(expr) => true,
                            _ => false,
                        };
                        let sugg_expr = if needs_parens { format!("({})", src) } else { src };

                        if let Some(sugg) = self.can_use_as_ref(expr) {
                            return Some((
                                sugg.0,
                                sugg.1,
                                sugg.2,
                                Applicability::MachineApplicable,
                            ));
                        }
                        let field_name = if is_struct_pat_shorthand_field {
                            format!("{}: ", sugg_expr)
                        } else {
                            String::new()
                        };
                        if let Some(hir::Node::Expr(hir::Expr {
                            kind: hir::ExprKind::Assign(left_expr, ..),
                            ..
                        })) = self.tcx.hir().find(self.tcx.hir().get_parent_node(expr.hir_id))
                        {
                            if mutability == hir::Mutability::Mut {
                                // Found the following case:
                                // fn foo(opt: &mut Option<String>){ opt = None }
                                //                                   ---   ^^^^
                                //                                   |     |
                                //    consider dereferencing here: `*opt`  |
                                // expected mutable reference, found enum `Option`
                                if let Ok(src) = sm.span_to_snippet(left_expr.span) {
                                    return Some((
                                        left_expr.span,
                                        "consider dereferencing here to assign to the mutable \
                                         borrowed piece of memory",
                                        format!("*{}", src),
                                        Applicability::MachineApplicable,
                                    ));
                                }
                            }
                        }

                        return Some(match mutability {
                            hir::Mutability::Mut => (
                                sp,
                                "consider mutably borrowing here",
                                format!("{}&mut {}", field_name, sugg_expr),
                                Applicability::MachineApplicable,
                            ),
                            hir::Mutability::Not => (
                                sp,
                                "consider borrowing here",
                                format!("{}&{}", field_name, sugg_expr),
                                Applicability::MachineApplicable,
                            ),
                        });
                    }
                }
            }
            (
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, ref expr),
                _,
                &ty::Ref(_, checked, _),
            ) if {
                self.infcx.can_sub(self.param_env, checked, &expected).is_ok() && !is_macro
            } =>
            {
                // We have `&T`, check if what was expected was `T`. If so,
                // we may want to suggest removing a `&`.
                if sm.is_imported(expr.span) {
                    if let Ok(src) = sm.span_to_snippet(sp) {
                        if let Some(src) = src.strip_prefix('&') {
                            return Some((
                                sp,
                                "consider removing the borrow",
                                src.to_string(),
                                Applicability::MachineApplicable,
                            ));
                        }
                    }
                    return None;
                }
                if let Ok(code) = sm.span_to_snippet(expr.span) {
                    return Some((
                        sp,
                        "consider removing the borrow",
                        code,
                        Applicability::MachineApplicable,
                    ));
                }
            }
            (
                _,
                &ty::RawPtr(TypeAndMut { ty: ty_b, mutbl: mutbl_b }),
                &ty::Ref(_, ty_a, mutbl_a),
            ) => {
                if let Some(steps) = self.deref_steps(ty_a, ty_b) {
                    // Only suggest valid if dereferencing needed.
                    if steps > 0 {
                        // The pointer type implements `Copy` trait so the suggestion is always valid.
                        if let Ok(src) = sm.span_to_snippet(sp) {
                            let derefs = &"*".repeat(steps);
                            if let Some((src, applicability)) = match mutbl_b {
                                hir::Mutability::Mut => {
                                    let new_prefix = "&mut ".to_owned() + derefs;
                                    match mutbl_a {
                                        hir::Mutability::Mut => {
                                            replace_prefix(&src, "&mut ", &new_prefix)
                                                .map(|s| (s, Applicability::MachineApplicable))
                                        }
                                        hir::Mutability::Not => {
                                            replace_prefix(&src, "&", &new_prefix)
                                                .map(|s| (s, Applicability::Unspecified))
                                        }
                                    }
                                }
                                hir::Mutability::Not => {
                                    let new_prefix = "&".to_owned() + derefs;
                                    match mutbl_a {
                                        hir::Mutability::Mut => {
                                            replace_prefix(&src, "&mut ", &new_prefix)
                                                .map(|s| (s, Applicability::MachineApplicable))
                                        }
                                        hir::Mutability::Not => {
                                            replace_prefix(&src, "&", &new_prefix)
                                                .map(|s| (s, Applicability::MachineApplicable))
                                        }
                                    }
                                }
                            } {
                                return Some((sp, "consider dereferencing", src, applicability));
                            }
                        }
                    }
                }
            }
            _ if sp == expr.span && !is_macro => {
                if let Some(steps) = self.deref_steps(checked_ty, expected) {
                    if steps == 1 {
                        // For a suggestion to make sense, the type would need to be `Copy`.
                        if self.infcx.type_is_copy_modulo_regions(self.param_env, expected, sp) {
                            if let Ok(code) = sm.span_to_snippet(sp) {
                                let message = if checked_ty.is_region_ptr() {
                                    "consider dereferencing the borrow"
                                } else {
                                    "consider dereferencing the type"
                                };
                                let suggestion = if is_struct_pat_shorthand_field {
                                    format!("{}: *{}", code, code)
                                } else {
                                    format!("*{}", code)
                                };
                                return Some((
                                    sp,
                                    message,
                                    suggestion,
                                    Applicability::MachineApplicable,
                                ));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        None
    }

    pub fn check_for_cast(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        checked_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) -> bool {
        if self.tcx.sess.source_map().is_imported(expr.span) {
            // Ignore if span is from within a macro.
            return false;
        }

        let src = if let Ok(src) = self.tcx.sess.source_map().span_to_snippet(expr.span) {
            src
        } else {
            return false;
        };

        // If casting this expression to a given numeric type would be appropriate in case of a type
        // mismatch.
        //
        // We want to minimize the amount of casting operations that are suggested, as it can be a
        // lossy operation with potentially bad side effects, so we only suggest when encountering
        // an expression that indicates that the original type couldn't be directly changed.
        //
        // For now, don't suggest casting with `as`.
        let can_cast = false;

        let prefix = if let Some(hir::Node::Expr(hir::Expr {
            kind: hir::ExprKind::Struct(_, fields, _),
            ..
        })) = self.tcx.hir().find(self.tcx.hir().get_parent_node(expr.hir_id))
        {
            // `expr` is a literal field for a struct, only suggest if appropriate
            match (*fields)
                .iter()
                .find(|field| field.expr.hir_id == expr.hir_id && field.is_shorthand)
            {
                // This is a field literal
                Some(field) => format!("{}: ", field.ident),
                // Likely a field was meant, but this field wasn't found. Do not suggest anything.
                None => return false,
            }
        } else {
            String::new()
        };

        if let hir::ExprKind::Call(path, args) = &expr.kind {
            if let (hir::ExprKind::Path(hir::QPath::TypeRelative(base_ty, path_segment)), 1) =
                (&path.kind, args.len())
            {
                // `expr` is a conversion like `u32::from(val)`, do not suggest anything (#63697).
                if let (hir::TyKind::Path(hir::QPath::Resolved(None, base_ty_path)), sym::from) =
                    (&base_ty.kind, path_segment.ident.name)
                {
                    if let Some(ident) = &base_ty_path.segments.iter().map(|s| s.ident).next() {
                        match ident.name {
                            sym::i128
                            | sym::i64
                            | sym::i32
                            | sym::i16
                            | sym::i8
                            | sym::u128
                            | sym::u64
                            | sym::u32
                            | sym::u16
                            | sym::u8
                            | sym::isize
                            | sym::usize
                                if base_ty_path.segments.len() == 1 =>
                            {
                                return false;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let msg = format!(
            "you can convert {} `{}` to {} `{}`",
            checked_ty.kind().article(),
            checked_ty,
            expected_ty.kind().article(),
            expected_ty,
        );
        let cast_msg = format!(
            "you can cast {} `{}` to {} `{}`",
            checked_ty.kind().article(),
            checked_ty,
            expected_ty.kind().article(),
            expected_ty,
        );
        let lit_msg = format!(
            "change the type of the numeric literal from `{}` to `{}`",
            checked_ty, expected_ty,
        );

        let with_opt_paren: fn(&dyn fmt::Display) -> String =
            if expr.precedence().order() < PREC_POSTFIX {
                |s| format!("({})", s)
            } else {
                |s| s.to_string()
            };

        let cast_suggestion = format!("{}{} as {}", prefix, with_opt_paren(&src), expected_ty);
        let into_suggestion = format!("{}{}.into()", prefix, with_opt_paren(&src));
        let suffix_suggestion = with_opt_paren(&format_args!(
            "{}{}",
            if matches!(
                (&expected_ty.kind(), &checked_ty.kind()),
                (ty::Int(_) | ty::Uint(_), ty::Float(_))
            ) {
                // Remove fractional part from literal, for example `42.0f32` into `42`
                let src = src.trim_end_matches(&checked_ty.to_string());
                src.split('.').next().unwrap()
            } else {
                src.trim_end_matches(&checked_ty.to_string())
            },
            expected_ty,
        ));
        let literal_is_ty_suffixed = |expr: &hir::Expr<'_>| {
            if let hir::ExprKind::Lit(lit) = &expr.kind { lit.node.is_suffixed() } else { false }
        };
        let is_negative_int =
            |expr: &hir::Expr<'_>| matches!(expr.kind, hir::ExprKind::Unary(hir::UnOp::UnNeg, ..));
        let is_uint = |ty: Ty<'_>| matches!(ty.kind(), ty::Uint(..));

        let in_const_context = self.tcx.hir().is_inside_const_context(expr.hir_id);

        let suggest_fallible_into_or_lhs_from =
            |err: &mut DiagnosticBuilder<'_>, exp_to_found_is_fallible: bool| {
                // If we know the expression the expected type is derived from, we might be able
                // to suggest a widening conversion rather than a narrowing one (which may
                // panic). For example, given x: u8 and y: u32, if we know the span of "x",
                //   x > y
                // can be given the suggestion "u32::from(x) > y" rather than
                // "x > y.try_into().unwrap()".
                let lhs_expr_and_src = expected_ty_expr.and_then(|expr| {
                    self.tcx
                        .sess
                        .source_map()
                        .span_to_snippet(expr.span)
                        .ok()
                        .map(|src| (expr, src))
                });
                let (span, msg, suggestion) = if let (Some((lhs_expr, lhs_src)), false) =
                    (lhs_expr_and_src, exp_to_found_is_fallible)
                {
                    let msg = format!(
                        "you can convert `{}` from `{}` to `{}`, matching the type of `{}`",
                        lhs_src, expected_ty, checked_ty, src
                    );
                    let suggestion = format!("{}::from({})", checked_ty, lhs_src);
                    (lhs_expr.span, msg, suggestion)
                } else {
                    let msg = format!("{} and panic if the converted value doesn't fit", msg);
                    let suggestion =
                        format!("{}{}.try_into().unwrap()", prefix, with_opt_paren(&src));
                    (expr.span, msg, suggestion)
                };
                err.span_suggestion(span, &msg, suggestion, Applicability::MachineApplicable);
            };

        let suggest_to_change_suffix_or_into =
            |err: &mut DiagnosticBuilder<'_>,
             found_to_exp_is_fallible: bool,
             exp_to_found_is_fallible: bool| {
                let always_fallible = found_to_exp_is_fallible
                    && (exp_to_found_is_fallible || expected_ty_expr.is_none());
                let msg = if literal_is_ty_suffixed(expr) {
                    &lit_msg
                } else if always_fallible && (is_negative_int(expr) && is_uint(expected_ty)) {
                    // We now know that converting either the lhs or rhs is fallible. Before we
                    // suggest a fallible conversion, check if the value can never fit in the
                    // expected type.
                    let msg = format!("`{}` cannot fit into type `{}`", src, expected_ty);
                    err.note(&msg);
                    return;
                } else if in_const_context {
                    // Do not recommend `into` or `try_into` in const contexts.
                    return;
                } else if found_to_exp_is_fallible {
                    return suggest_fallible_into_or_lhs_from(err, exp_to_found_is_fallible);
                } else {
                    &msg
                };
                let suggestion = if literal_is_ty_suffixed(expr) {
                    suffix_suggestion.clone()
                } else {
                    into_suggestion.clone()
                };
                err.span_suggestion(expr.span, msg, suggestion, Applicability::MachineApplicable);
            };

        match (&expected_ty.kind(), &checked_ty.kind()) {
            (&ty::Int(ref exp), &ty::Int(ref found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if exp < found => (true, false),
                    (Some(exp), Some(found)) if exp > found => (false, true),
                    (None, Some(8 | 16)) => (false, true),
                    (Some(8 | 16), None) => (true, false),
                    (None, _) | (_, None) => (true, true),
                    _ => (false, false),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (&ty::Uint(ref exp), &ty::Uint(ref found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if exp < found => (true, false),
                    (Some(exp), Some(found)) if exp > found => (false, true),
                    (None, Some(8 | 16)) => (false, true),
                    (Some(8 | 16), None) => (true, false),
                    (None, _) | (_, None) => (true, true),
                    _ => (false, false),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (&ty::Int(exp), &ty::Uint(found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if found < exp => (false, true),
                    (None, Some(8)) => (false, true),
                    _ => (true, true),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (&ty::Uint(exp), &ty::Int(found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if found > exp => (true, false),
                    (Some(8), None) => (true, false),
                    _ => (true, true),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (&ty::Float(ref exp), &ty::Float(ref found)) => {
                if found.bit_width() < exp.bit_width() {
                    suggest_to_change_suffix_or_into(err, false, true);
                } else if literal_is_ty_suffixed(expr) {
                    err.span_suggestion(
                        expr.span,
                        &lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if can_cast {
                    // Missing try_into implementation for `f64` to `f32`
                    err.span_suggestion(
                        expr.span,
                        &format!("{}, producing the closest possible value", cast_msg),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            (&ty::Uint(_) | &ty::Int(_), &ty::Float(_)) => {
                if literal_is_ty_suffixed(expr) {
                    err.span_suggestion(
                        expr.span,
                        &lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if can_cast {
                    // Missing try_into implementation for `{float}` to `{integer}`
                    err.span_suggestion(
                        expr.span,
                        &format!("{}, rounding the float towards zero", msg),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            (&ty::Float(ref exp), &ty::Uint(ref found)) => {
                // if `found` is `None` (meaning found is `usize`), don't suggest `.into()`
                if exp.bit_width() > found.bit_width().unwrap_or(256) {
                    err.span_suggestion(
                        expr.span,
                        &format!(
                            "{}, producing the floating point representation of the integer",
                            msg,
                        ),
                        into_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if literal_is_ty_suffixed(expr) {
                    err.span_suggestion(
                        expr.span,
                        &lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else {
                    // Missing try_into implementation for `{integer}` to `{float}`
                    err.span_suggestion(
                        expr.span,
                        &format!(
                            "{}, producing the floating point representation of the integer,
                                 rounded if necessary",
                            cast_msg,
                        ),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            (&ty::Float(ref exp), &ty::Int(ref found)) => {
                // if `found` is `None` (meaning found is `isize`), don't suggest `.into()`
                if exp.bit_width() > found.bit_width().unwrap_or(256) {
                    err.span_suggestion(
                        expr.span,
                        &format!(
                            "{}, producing the floating point representation of the integer",
                            &msg,
                        ),
                        into_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if literal_is_ty_suffixed(expr) {
                    err.span_suggestion(
                        expr.span,
                        &lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else {
                    // Missing try_into implementation for `{integer}` to `{float}`
                    err.span_suggestion(
                        expr.span,
                        &format!(
                            "{}, producing the floating point representation of the integer, \
                                rounded if necessary",
                            &msg,
                        ),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            _ => false,
        }
    }
}

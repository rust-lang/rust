use crate::check::FnCtxt;
use rustc::infer::InferOk;
use rustc::traits::{ObligationCause, ObligationCauseCode};

use syntax::util::parser::PREC_POSTFIX;
use syntax_pos::Span;
use rustc::hir;
use rustc::hir::def::Def;
use rustc::hir::Node;
use rustc::hir::{Item, ItemKind, print};
use rustc::ty::{self, Ty, AssociatedItem};
use rustc::ty::adjustment::AllowTwoPhase;
use errors::{Applicability, DiagnosticBuilder};

use super::method::probe;

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    // Requires that the two types unify, and prints an error message if
    // they don't.
    pub fn demand_suptype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        self.demand_suptype_diag(sp, expected, actual).map(|mut e| e.emit());
    }

    pub fn demand_suptype_diag(&self,
                               sp: Span,
                               expected: Ty<'tcx>,
                               actual: Ty<'tcx>) -> Option<DiagnosticBuilder<'tcx>> {
        let cause = &self.misc(sp);
        match self.at(cause, self.param_env).sup(expected, actual) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
                None
            },
            Err(e) => {
                Some(self.report_mismatched_types(&cause, expected, actual, e))
            }
        }
    }

    pub fn demand_eqtype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        if let Some(mut err) = self.demand_eqtype_diag(sp, expected, actual) {
            err.emit();
        }
    }

    pub fn demand_eqtype_diag(&self,
                             sp: Span,
                             expected: Ty<'tcx>,
                             actual: Ty<'tcx>) -> Option<DiagnosticBuilder<'tcx>> {
        self.demand_eqtype_with_origin(&self.misc(sp), expected, actual)
    }

    pub fn demand_eqtype_with_origin(&self,
                                     cause: &ObligationCause<'tcx>,
                                     expected: Ty<'tcx>,
                                     actual: Ty<'tcx>) -> Option<DiagnosticBuilder<'tcx>> {
        match self.at(cause, self.param_env).eq(expected, actual) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
                None
            }
            Err(e) => {
                Some(self.report_mismatched_types(cause, expected, actual, e))
            }
        }
    }

    pub fn demand_eqtype_pat(
        &self,
        cause_span: Span,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        match_expr_span: Option<Span>,
    ) {
        let cause = if let Some(span) = match_expr_span {
            self.cause(
                cause_span,
                ObligationCauseCode::MatchExpressionArmPattern { span, ty: expected },
            )
        } else {
            self.misc(cause_span)
        };
        self.demand_eqtype_with_origin(&cause, expected, actual).map(|mut err| err.emit());
    }


    pub fn demand_coerce(&self,
                         expr: &hir::Expr,
                         checked_ty: Ty<'tcx>,
                         expected: Ty<'tcx>,
                         allow_two_phase: AllowTwoPhase)
                         -> Ty<'tcx> {
        let (ty, err) = self.demand_coerce_diag(expr, checked_ty, expected, allow_two_phase);
        if let Some(mut err) = err {
            err.emit();
        }
        ty
    }

    // Checks that the type of `expr` can be coerced to `expected`.
    //
    // N.B., this code relies on `self.diverges` to be accurate. In
    // particular, assignments to `!` will be permitted if the
    // diverges flag is currently "always".
    pub fn demand_coerce_diag(&self,
                              expr: &hir::Expr,
                              checked_ty: Ty<'tcx>,
                              expected: Ty<'tcx>,
                              allow_two_phase: AllowTwoPhase)
                              -> (Ty<'tcx>, Option<DiagnosticBuilder<'tcx>>) {
        let expected = self.resolve_type_vars_with_obligations(expected);

        let e = match self.try_coerce(expr, checked_ty, expected, allow_two_phase) {
            Ok(ty) => return (ty, None),
            Err(e) => e
        };

        let cause = self.misc(expr.span);
        let expr_ty = self.resolve_type_vars_with_obligations(checked_ty);
        let mut err = self.report_mismatched_types(&cause, expected, expr_ty, e);

        if self.is_assign_to_bool(expr, expected) {
            // Error reported in `check_assign` so avoid emitting error again.
            err.delay_as_bug();
            return (expected, None)
        }

        self.suggest_compatible_variants(&mut err, expr, expected, expr_ty);
        self.suggest_ref_or_into(&mut err, expr, expected, expr_ty);

        (expected, Some(err))
    }

    /// Returns whether the expected type is `bool` and the expression is `x = y`.
    pub fn is_assign_to_bool(&self, expr: &hir::Expr, expected: Ty<'tcx>) -> bool {
        if let hir::ExprKind::Assign(..) = expr.node {
            return expected == self.tcx.types.bool;
        }
        false
    }

    /// If the expected type is an enum (Issue #55250) with any variants whose
    /// sole field is of the found type, suggest such variants. (Issue #42764)
    fn suggest_compatible_variants(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr,
        expected: Ty<'tcx>,
        expr_ty: Ty<'tcx>,
    ) {
        if let ty::Adt(expected_adt, substs) = expected.sty {
            if !expected_adt.is_enum() {
                return;
            }

            let mut compatible_variants = expected_adt.variants
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
                }).peekable();

            if compatible_variants.peek().is_some() {
                let expr_text = print::to_string(print::NO_ANN, |s| s.print_expr(expr));
                let suggestions = compatible_variants
                    .map(|v| format!("{}({})", v, expr_text));
                let msg = "try using a variant of the expected type";
                err.span_suggestions(expr.span, msg, suggestions, Applicability::MaybeIncorrect);
            }
        }
    }

    pub fn get_conversion_methods(&self, span: Span, expected: Ty<'tcx>, checked_ty: Ty<'tcx>)
                              -> Vec<AssociatedItem> {
        let mut methods = self.probe_for_return_type(span,
                                                     probe::Mode::MethodCall,
                                                     expected,
                                                     checked_ty,
                                                     hir::DUMMY_HIR_ID);
        methods.retain(|m| {
            self.has_no_input_arg(m) &&
                self.tcx.get_attrs(m.def_id).iter()
                // This special internal attribute is used to whitelist
                // "identity-like" conversion methods to be suggested here.
                //
                // FIXME (#46459 and #46460): ideally
                // `std::convert::Into::into` and `std::borrow:ToOwned` would
                // also be `#[rustc_conversion_suggestion]`, if not for
                // method-probing false-positives and -negatives (respectively).
                //
                // FIXME? Other potential candidate methods: `as_ref` and
                // `as_mut`?
                .find(|a| a.check_name("rustc_conversion_suggestion")).is_some()
        });

        methods
    }

    // This function checks if the method isn't static and takes other arguments than `self`.
    fn has_no_input_arg(&self, method: &AssociatedItem) -> bool {
        match method.def() {
            Def::Method(def_id) => {
                self.tcx.fn_sig(def_id).inputs().skip_binder().len() == 1
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
    /// opt.map(|arg| takes_ref(arg));
    /// ```
    /// Suggest using `opt.as_ref().map(|arg| takes_ref(arg));` instead.
    ///
    /// It only checks for `Option` and `Result` and won't work with
    /// ```
    /// opt.map(|arg| { takes_ref(arg) });
    /// ```
    fn can_use_as_ref(
        &self,
        expr: &hir::Expr,
    ) -> Option<(Span, &'static str, String)> {
        if let hir::ExprKind::Path(hir::QPath::Resolved(_, ref path)) = expr.node {
            if let hir::def::Def::Local(id) = path.def {
                let parent = self.tcx.hir().get_parent_node_by_hir_id(id);
                if let Some(Node::Expr(hir::Expr {
                    hir_id,
                    node: hir::ExprKind::Closure(_, decl, ..),
                    ..
                })) = self.tcx.hir().find_by_hir_id(parent) {
                    let parent = self.tcx.hir().get_parent_node_by_hir_id(*hir_id);
                    if let (Some(Node::Expr(hir::Expr {
                        node: hir::ExprKind::MethodCall(path, span, expr),
                        ..
                    })), 1) = (self.tcx.hir().find_by_hir_id(parent), decl.inputs.len()) {
                        let self_ty = self.tables.borrow().node_type(expr[0].hir_id);
                        let self_ty = format!("{:?}", self_ty);
                        let name = path.ident.as_str();
                        let is_as_ref_able = (
                            self_ty.starts_with("&std::option::Option") ||
                            self_ty.starts_with("&std::result::Result") ||
                            self_ty.starts_with("std::option::Option") ||
                            self_ty.starts_with("std::result::Result")
                        ) && (name == "map" || name == "and_then");
                        match (is_as_ref_able, self.sess().source_map().span_to_snippet(*span)) {
                            (true, Ok(src)) => {
                                return Some((*span, "consider using `as_ref` instead",
                                             format!("as_ref().{}", src)));
                            },
                            _ => ()
                        }
                    }
                }
            }
        }
        None
    }

    fn is_hir_id_from_struct_pattern_shorthand_field(&self, hir_id: hir::HirId, sp: Span) -> bool {
        let cm = self.sess().source_map();
        let parent_id = self.tcx.hir().get_parent_node_by_hir_id(hir_id);
        if let Some(parent) = self.tcx.hir().find_by_hir_id(parent_id) {
            // Account for fields
            if let Node::Expr(hir::Expr {
                node: hir::ExprKind::Struct(_, fields, ..), ..
            }) = parent {
                if let Ok(src) = cm.span_to_snippet(sp) {
                    for field in fields {
                        if field.ident.as_str() == src.as_str() && field.is_shorthand {
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
    pub fn check_ref(&self,
                 expr: &hir::Expr,
                 checked_ty: Ty<'tcx>,
                 expected: Ty<'tcx>)
                 -> Option<(Span, &'static str, String)> {
        let cm = self.sess().source_map();
        let sp = expr.span;
        if !cm.span_to_filename(sp).is_real() {
            // Ignore if span is from within a macro #41858, #58298. We previously used the macro
            // call span, but that breaks down when the type error comes from multiple calls down.
            return None;
        }

        let is_struct_pat_shorthand_field = self.is_hir_id_from_struct_pattern_shorthand_field(
            expr.hir_id,
            sp,
        );

        match (&expected.sty, &checked_ty.sty) {
            (&ty::Ref(_, exp, _), &ty::Ref(_, check, _)) => match (&exp.sty, &check.sty) {
                (&ty::Str, &ty::Array(arr, _)) |
                (&ty::Str, &ty::Slice(arr)) if arr == self.tcx.types.u8 => {
                    if let hir::ExprKind::Lit(_) = expr.node {
                        if let Ok(src) = cm.span_to_snippet(sp) {
                            if src.starts_with("b\"") {
                                return Some((sp,
                                             "consider removing the leading `b`",
                                             src[1..].to_string()));
                            }
                        }
                    }
                },
                (&ty::Array(arr, _), &ty::Str) |
                (&ty::Slice(arr), &ty::Str) if arr == self.tcx.types.u8 => {
                    if let hir::ExprKind::Lit(_) = expr.node {
                        if let Ok(src) = cm.span_to_snippet(sp) {
                            if src.starts_with("\"") {
                                return Some((sp,
                                             "consider adding a leading `b`",
                                             format!("b{}", src)));
                            }
                        }
                    }
                }
                _ => {}
            },
            (&ty::Ref(_, _, mutability), _) => {
                // Check if it can work when put into a ref. For example:
                //
                // ```
                // fn bar(x: &mut i32) {}
                //
                // let x = 0u32;
                // bar(&x); // error, expected &mut
                // ```
                let ref_ty = match mutability {
                    hir::Mutability::MutMutable => {
                        self.tcx.mk_mut_ref(self.tcx.mk_region(ty::ReStatic), checked_ty)
                    }
                    hir::Mutability::MutImmutable => {
                        self.tcx.mk_imm_ref(self.tcx.mk_region(ty::ReStatic), checked_ty)
                    }
                };
                if self.can_coerce(ref_ty, expected) {
                    if let Ok(src) = cm.span_to_snippet(sp) {
                        let needs_parens = match expr.node {
                            // parenthesize if needed (Issue #46756)
                            hir::ExprKind::Cast(_, _) |
                            hir::ExprKind::Binary(_, _, _) => true,
                            // parenthesize borrows of range literals (Issue #54505)
                            _ if self.is_range_literal(expr) => true,
                            _ => false,
                        };
                        let sugg_expr = if needs_parens {
                            format!("({})", src)
                        } else {
                            src
                        };

                        if let Some(sugg) = self.can_use_as_ref(expr) {
                            return Some(sugg);
                        }
                        let field_name = if is_struct_pat_shorthand_field {
                            format!("{}: ", sugg_expr)
                        } else {
                            String::new()
                        };
                        return Some(match mutability {
                            hir::Mutability::MutMutable => (
                                sp,
                                "consider mutably borrowing here",
                                format!("{}&mut {}", field_name, sugg_expr),
                            ),
                            hir::Mutability::MutImmutable => (
                                sp,
                                "consider borrowing here",
                                format!("{}&{}", field_name, sugg_expr),
                            ),
                        });
                    }
                }
            }
            (_, &ty::Ref(_, checked, _)) => {
                // We have `&T`, check if what was expected was `T`. If so,
                // we may want to suggest adding a `*`, or removing
                // a `&`.
                //
                // (But, also check the `expn_info()` to see if this is
                // a macro; if so, it's hard to extract the text and make a good
                // suggestion, so don't bother.)
                if self.infcx.can_sub(self.param_env, checked, &expected).is_ok() &&
                   sp.ctxt().outer().expn_info().is_none() {
                    match expr.node {
                        // Maybe remove `&`?
                        hir::ExprKind::AddrOf(_, ref expr) => {
                            if !cm.span_to_filename(expr.span).is_real() {
                                if let Ok(code) = cm.span_to_snippet(sp) {
                                    if code.chars().next() == Some('&') {
                                        return Some((
                                            sp,
                                            "consider removing the borrow",
                                            code[1..].to_string()),
                                        );
                                    }
                                }
                                return None;
                            }
                            if let Ok(code) = cm.span_to_snippet(expr.span) {
                                return Some((sp, "consider removing the borrow", code));
                            }
                        }

                        // Maybe add `*`? Only if `T: Copy`.
                        _ => {
                            if self.infcx.type_is_copy_modulo_regions(self.param_env,
                                                                      checked,
                                                                      sp) {
                                // do not suggest if the span comes from a macro (#52783)
                                if let (Ok(code), true) = (
                                    cm.span_to_snippet(sp),
                                    sp == expr.span,
                                ) {
                                    return Some((
                                        sp,
                                        "consider dereferencing the borrow",
                                        if is_struct_pat_shorthand_field {
                                            format!("{}: *{}", code, code)
                                        } else {
                                            format!("*{}", code)
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// This function checks if the specified expression is a built-in range literal.
    /// (See: `LoweringContext::lower_expr()` in `src/librustc/hir/lowering.rs`).
    fn is_range_literal(&self, expr: &hir::Expr) -> bool {
        use hir::{Path, QPath, ExprKind, TyKind};

        // We support `::std::ops::Range` and `::core::ops::Range` prefixes
        let is_range_path = |path: &Path| {
            let mut segs = path.segments.iter()
                .map(|seg| seg.ident.as_str());

            if let (Some(root), Some(std_core), Some(ops), Some(range), None) =
                (segs.next(), segs.next(), segs.next(), segs.next(), segs.next())
            {
                // "{{root}}" is the equivalent of `::` prefix in Path
                root == "{{root}}" && (std_core == "std" || std_core == "core")
                    && ops == "ops" && range.starts_with("Range")
            } else {
                false
            }
        };

        let span_is_range_literal = |span: &Span| {
            // Check whether a span corresponding to a range expression
            // is a range literal, rather than an explicit struct or `new()` call.
            let source_map = self.tcx.sess.source_map();
            let end_point = source_map.end_point(*span);

            if let Ok(end_string) = source_map.span_to_snippet(end_point) {
                !(end_string.ends_with("}") || end_string.ends_with(")"))
            } else {
                false
            }
        };

        match expr.node {
            // All built-in range literals but `..=` and `..` desugar to Structs
            ExprKind::Struct(ref qpath, _, _) => {
                if let QPath::Resolved(None, ref path) = **qpath {
                    return is_range_path(&path) && span_is_range_literal(&expr.span);
                }
            }
            // `..` desugars to its struct path
            ExprKind::Path(QPath::Resolved(None, ref path)) => {
                return is_range_path(&path) && span_is_range_literal(&expr.span);
            }

            // `..=` desugars into `::std::ops::RangeInclusive::new(...)`
            ExprKind::Call(ref func, _) => {
                if let ExprKind::Path(QPath::TypeRelative(ref ty, ref segment)) = func.node {
                    if let TyKind::Path(QPath::Resolved(None, ref path)) = ty.node {
                        let call_to_new = segment.ident.as_str() == "new";

                        return is_range_path(&path) && span_is_range_literal(&expr.span)
                            && call_to_new;
                    }
                }
            }

            _ => {}
        }

        false
    }

    pub fn check_for_cast(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        expr: &hir::Expr,
        checked_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        let parent_id = self.tcx.hir().get_parent_node_by_hir_id(expr.hir_id);
        if let Some(parent) = self.tcx.hir().find_by_hir_id(parent_id) {
            // Shouldn't suggest `.into()` on `const`s.
            if let Node::Item(Item { node: ItemKind::Const(_, _), .. }) = parent {
                // FIXME(estebank): modify once we decide to suggest `as` casts
                return false;
            }
        };

        let will_truncate = "will truncate the source value";
        let depending_on_isize = "will truncate or zero-extend depending on the bit width of \
                                  `isize`";
        let depending_on_usize = "will truncate or zero-extend depending on the bit width of \
                                  `usize`";
        let will_sign_extend = "will sign-extend the source value";
        let will_zero_extend = "will zero-extend the source value";

        // If casting this expression to a given numeric type would be appropriate in case of a type
        // mismatch.
        //
        // We want to minimize the amount of casting operations that are suggested, as it can be a
        // lossy operation with potentially bad side effects, so we only suggest when encountering
        // an expression that indicates that the original type couldn't be directly changed.
        //
        // For now, don't suggest casting with `as`.
        let can_cast = false;

        let mut prefix = String::new();
        if let Some(hir::Node::Expr(hir::Expr {
            node: hir::ExprKind::Struct(_, fields, _),
            ..
        })) = self.tcx.hir().find_by_hir_id(self.tcx.hir().get_parent_node_by_hir_id(expr.hir_id)) {
            // `expr` is a literal field for a struct, only suggest if appropriate
            for field in fields {
                if field.expr.hir_id == expr.hir_id && field.is_shorthand {
                    // This is a field literal
                    prefix = format!("{}: ", field.ident);
                    break;
                }
            }
            if &prefix == "" {
                // Likely a field was meant, but this field wasn't found. Do not suggest anything.
                return false;
            }
        }

        let needs_paren = expr.precedence().order() < (PREC_POSTFIX as i8);

        if let Ok(src) = self.tcx.sess.source_map().span_to_snippet(expr.span) {
            let msg = format!("you can cast an `{}` to `{}`", checked_ty, expected_ty);
            let cast_suggestion = format!(
                "{}{}{}{} as {}",
                prefix,
                if needs_paren { "(" } else { "" },
                src,
                if needs_paren { ")" } else { "" },
                expected_ty,
            );
            let into_suggestion = format!(
                "{}{}{}{}.into()",
                prefix,
                if needs_paren { "(" } else { "" },
                src,
                if needs_paren { ")" } else { "" },
            );
            let literal_is_ty_suffixed = |expr: &hir::Expr| {
                if let hir::ExprKind::Lit(lit) = &expr.node {
                    lit.node.is_suffixed()
                } else {
                    false
                }
            };

            let into_sugg = into_suggestion.clone();
            let suggest_to_change_suffix_or_into = |err: &mut DiagnosticBuilder<'_>,
                                                    note: Option<&str>| {
                let suggest_msg = if literal_is_ty_suffixed(expr) {
                    format!(
                        "change the type of the numeric literal from `{}` to `{}`",
                        checked_ty,
                        expected_ty,
                    )
                } else {
                    match note {
                        Some(note) => format!("{}, which {}", msg, note),
                        _ => format!("{} in a lossless way", msg),
                    }
                };

                let suffix_suggestion = format!(
                    "{}{}{}{}",
                    if needs_paren { "(" } else { "" },
                    src.trim_end_matches(&checked_ty.to_string()),
                    expected_ty,
                    if needs_paren { ")" } else { "" },
                );

                err.span_suggestion(
                    expr.span,
                    &suggest_msg,
                    if literal_is_ty_suffixed(expr) {
                        suffix_suggestion
                    } else {
                        into_sugg
                    },
                    Applicability::MachineApplicable,
                );
            };

            match (&expected_ty.sty, &checked_ty.sty) {
                (&ty::Int(ref exp), &ty::Int(ref found)) => {
                    match (found.bit_width(), exp.bit_width()) {
                        (Some(found), Some(exp)) if found > exp => {
                            if can_cast {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_truncate),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect // lossy conversion
                                );
                            }
                        }
                        (None, _) | (_, None) => {
                            if can_cast {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, depending_on_isize),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect // lossy conversion
                                );
                            }
                        }
                        _ => {
                            suggest_to_change_suffix_or_into(
                                err,
                                Some(will_sign_extend),
                            );
                        }
                    }
                    true
                }
                (&ty::Uint(ref exp), &ty::Uint(ref found)) => {
                    match (found.bit_width(), exp.bit_width()) {
                        (Some(found), Some(exp)) if found > exp => {
                            if can_cast {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_truncate),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                        }
                        (None, _) | (_, None) => {
                            if can_cast {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, depending_on_usize),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                        }
                        _ => {
                           suggest_to_change_suffix_or_into(
                               err,
                               Some(will_zero_extend),
                           );
                        }
                    }
                    true
                }
                (&ty::Int(ref exp), &ty::Uint(ref found)) => {
                    if can_cast {
                        match (found.bit_width(), exp.bit_width()) {
                            (Some(found), Some(exp)) if found > exp - 1 => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_truncate),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                            (None, None) => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_truncate),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                            (None, _) => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, depending_on_isize),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                            (_, None) => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, depending_on_usize),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                            _ => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_zero_extend),
                                    cast_suggestion,
                                    Applicability::MachineApplicable
                                );
                            }
                        }
                    }
                    true
                }
                (&ty::Uint(ref exp), &ty::Int(ref found)) => {
                    if can_cast {
                        match (found.bit_width(), exp.bit_width()) {
                            (Some(found), Some(exp)) if found - 1 > exp => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_truncate),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                            (None, None) => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_sign_extend),
                                    cast_suggestion,
                                    Applicability::MachineApplicable  // lossy conversion
                                );
                            }
                            (None, _) => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, depending_on_usize),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                            (_, None) => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, depending_on_isize),
                                    cast_suggestion,
                                    Applicability::MaybeIncorrect  // lossy conversion
                                );
                            }
                            _ => {
                                err.span_suggestion(
                                    expr.span,
                                    &format!("{}, which {}", msg, will_sign_extend),
                                    cast_suggestion,
                                    Applicability::MachineApplicable
                                );
                            }
                        }
                    }
                    true
                }
                (&ty::Float(ref exp), &ty::Float(ref found)) => {
                    if found.bit_width() < exp.bit_width() {
                       suggest_to_change_suffix_or_into(
                           err,
                           None,
                       );
                    } else if can_cast {
                        err.span_suggestion(
                            expr.span,
                            &format!("{}, producing the closest possible value", msg),
                            cast_suggestion,
                            Applicability::MaybeIncorrect  // lossy conversion
                        );
                    }
                    true
                }
                (&ty::Uint(_), &ty::Float(_)) | (&ty::Int(_), &ty::Float(_)) => {
                    if can_cast {
                        err.span_suggestion(
                            expr.span,
                            &format!("{}, rounding the float towards zero", msg),
                            cast_suggestion,
                            Applicability::MaybeIncorrect  // lossy conversion
                        );
                        err.warn("casting here will cause undefined behavior if the rounded value \
                                  cannot be represented by the target integer type, including \
                                  `Inf` and `NaN` (this is a bug and will be fixed)");
                    }
                    true
                }
                (&ty::Float(ref exp), &ty::Uint(ref found)) => {
                    // if `found` is `None` (meaning found is `usize`), don't suggest `.into()`
                    if exp.bit_width() > found.bit_width().unwrap_or(256) {
                        err.span_suggestion(
                            expr.span,
                            &format!("{}, producing the floating point representation of the \
                                      integer",
                                     msg),
                            into_suggestion,
                            Applicability::MachineApplicable
                        );
                    } else if can_cast {
                        err.span_suggestion(
                            expr.span,
                            &format!("{}, producing the floating point representation of the \
                                      integer, rounded if necessary",
                                     msg),
                            cast_suggestion,
                            Applicability::MaybeIncorrect  // lossy conversion
                        );
                    }
                    true
                }
                (&ty::Float(ref exp), &ty::Int(ref found)) => {
                    // if `found` is `None` (meaning found is `isize`), don't suggest `.into()`
                    if exp.bit_width() > found.bit_width().unwrap_or(256) {
                        err.span_suggestion(
                            expr.span,
                            &format!("{}, producing the floating point representation of the \
                                      integer",
                                     msg),
                            into_suggestion,
                            Applicability::MachineApplicable
                        );
                    } else if can_cast {
                        err.span_suggestion(
                            expr.span,
                            &format!("{}, producing the floating point representation of the \
                                      integer, rounded if necessary",
                                     msg),
                            cast_suggestion,
                            Applicability::MaybeIncorrect  // lossy conversion
                        );
                    }
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }
}

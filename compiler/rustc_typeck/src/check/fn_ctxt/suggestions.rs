use super::FnCtxt;
use crate::astconv::AstConv;

use rustc_ast::util::parser::ExprPrecedence;
use rustc_span::{self, Span};

use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ExprKind, ItemKind, Node};
use rustc_infer::infer;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::kw;

use std::iter;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(in super::super) fn suggest_semicolon_at_end(
        &self,
        span: Span,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        err.span_suggestion_short(
            span.shrink_to_hi(),
            "consider using a semicolon here",
            ";".to_string(),
            Applicability::MachineApplicable,
        );
    }

    /// On implicit return expressions with mismatched types, provides the following suggestions:
    ///
    /// - Points out the method's return type as the reason for the expected type.
    /// - Possible missing semicolon.
    /// - Possible missing return type if the return type is the default, and not `fn main()`.
    pub fn suggest_mismatched_types_on_tail(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        cause_span: Span,
        blk_id: hir::HirId,
    ) -> bool {
        let expr = expr.peel_drop_temps();
        self.suggest_missing_semicolon(err, expr, expected, cause_span);
        let mut pointing_at_return_type = false;
        if let Some((fn_decl, can_suggest)) = self.get_fn_decl(blk_id) {
            pointing_at_return_type =
                self.suggest_missing_return_type(err, &fn_decl, expected, found, can_suggest);
        }
        pointing_at_return_type
    }

    /// When encountering an fn-like ctor that needs to unify with a value, check whether calling
    /// the ctor would successfully solve the type mismatch and if so, suggest it:
    /// ```
    /// fn foo(x: usize) -> usize { x }
    /// let x: usize = foo;  // suggest calling the `foo` function: `foo(42)`
    /// ```
    fn suggest_fn_call(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        let hir = self.tcx.hir();
        let (def_id, sig) = match *found.kind() {
            ty::FnDef(def_id, _) => (def_id, found.fn_sig(self.tcx)),
            ty::Closure(def_id, substs) => (def_id, substs.as_closure().sig()),
            _ => return false,
        };

        let sig = self.replace_bound_vars_with_fresh_vars(expr.span, infer::FnCall, sig).0;
        let sig = self.normalize_associated_types_in(expr.span, sig);
        if self.can_coerce(sig.output(), expected) {
            let (mut sugg_call, applicability) = if sig.inputs().is_empty() {
                (String::new(), Applicability::MachineApplicable)
            } else {
                ("...".to_string(), Applicability::HasPlaceholders)
            };
            let mut msg = "call this function";
            match hir.get_if_local(def_id) {
                Some(
                    Node::Item(hir::Item { kind: ItemKind::Fn(.., body_id), .. })
                    | Node::ImplItem(hir::ImplItem {
                        kind: hir::ImplItemKind::Fn(_, body_id), ..
                    })
                    | Node::TraitItem(hir::TraitItem {
                        kind: hir::TraitItemKind::Fn(.., hir::TraitFn::Provided(body_id)),
                        ..
                    }),
                ) => {
                    let body = hir.body(*body_id);
                    sugg_call = body
                        .params
                        .iter()
                        .map(|param| match &param.pat.kind {
                            hir::PatKind::Binding(_, _, ident, None)
                                if ident.name != kw::SelfLower =>
                            {
                                ident.to_string()
                            }
                            _ => "_".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                Some(Node::Expr(hir::Expr {
                    kind: ExprKind::Closure(_, _, body_id, _, _),
                    span: full_closure_span,
                    ..
                })) => {
                    if *full_closure_span == expr.span {
                        return false;
                    }
                    msg = "call this closure";
                    let body = hir.body(*body_id);
                    sugg_call = body
                        .params
                        .iter()
                        .map(|param| match &param.pat.kind {
                            hir::PatKind::Binding(_, _, ident, None)
                                if ident.name != kw::SelfLower =>
                            {
                                ident.to_string()
                            }
                            _ => "_".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                Some(Node::Ctor(hir::VariantData::Tuple(fields, _))) => {
                    sugg_call = fields.iter().map(|_| "_").collect::<Vec<_>>().join(", ");
                    match def_id.as_local().map(|def_id| hir.def_kind(def_id)) {
                        Some(DefKind::Ctor(hir::def::CtorOf::Variant, _)) => {
                            msg = "instantiate this tuple variant";
                        }
                        Some(DefKind::Ctor(CtorOf::Struct, _)) => {
                            msg = "instantiate this tuple struct";
                        }
                        _ => {}
                    }
                }
                Some(Node::ForeignItem(hir::ForeignItem {
                    kind: hir::ForeignItemKind::Fn(_, idents, _),
                    ..
                })) => {
                    sugg_call = idents
                        .iter()
                        .map(|ident| {
                            if ident.name != kw::SelfLower {
                                ident.to_string()
                            } else {
                                "_".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                Some(Node::TraitItem(hir::TraitItem {
                    kind: hir::TraitItemKind::Fn(.., hir::TraitFn::Required(idents)),
                    ..
                })) => {
                    sugg_call = idents
                        .iter()
                        .map(|ident| {
                            if ident.name != kw::SelfLower {
                                ident.to_string()
                            } else {
                                "_".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                _ => {}
            }
            err.span_suggestion_verbose(
                expr.span.shrink_to_hi(),
                &format!("use parentheses to {}", msg),
                format!("({})", sugg_call),
                applicability,
            );
            return true;
        }
        false
    }

    pub fn suggest_deref_ref_or_into(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) {
        if let Some((sp, msg, suggestion, applicability)) = self.check_ref(expr, found, expected) {
            err.span_suggestion(sp, msg, suggestion, applicability);
        } else if let (ty::FnDef(def_id, ..), true) =
            (&found.kind(), self.suggest_fn_call(err, expr, expected, found))
        {
            if let Some(sp) = self.tcx.hir().span_if_local(*def_id) {
                let sp = self.sess().source_map().guess_head_span(sp);
                err.span_label(sp, &format!("{} defined here", found));
            }
        } else if !self.check_for_cast(err, expr, found, expected, expected_ty_expr) {
            let is_struct_pat_shorthand_field =
                self.is_hir_id_from_struct_pattern_shorthand_field(expr.hir_id, expr.span);
            let methods = self.get_conversion_methods(expr.span, expected, found, expr.hir_id);
            if let Ok(expr_text) = self.sess().source_map().span_to_snippet(expr.span) {
                let mut suggestions = iter::repeat(&expr_text)
                    .zip(methods.iter())
                    .filter_map(|(receiver, method)| {
                        let method_call = format!(".{}()", method.ident);
                        if receiver.ends_with(&method_call) {
                            None // do not suggest code that is already there (#53348)
                        } else {
                            let method_call_list = [".to_vec()", ".to_string()"];
                            let sugg = if receiver.ends_with(".clone()")
                                && method_call_list.contains(&method_call.as_str())
                            {
                                let max_len = receiver.rfind('.').unwrap();
                                format!("{}{}", &receiver[..max_len], method_call)
                            } else {
                                if expr.precedence().order() < ExprPrecedence::MethodCall.order() {
                                    format!("({}){}", receiver, method_call)
                                } else {
                                    format!("{}{}", receiver, method_call)
                                }
                            };
                            Some(if is_struct_pat_shorthand_field {
                                format!("{}: {}", receiver, sugg)
                            } else {
                                sugg
                            })
                        }
                    })
                    .peekable();
                if suggestions.peek().is_some() {
                    err.span_suggestions(
                        expr.span,
                        "try using a conversion method",
                        suggestions,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    /// When encountering the expected boxed value allocated in the stack, suggest allocating it
    /// in the heap by calling `Box::new()`.
    pub(in super::super) fn suggest_boxing_when_appropriate(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        if self.tcx.hir().is_inside_const_context(expr.hir_id) {
            // Do not suggest `Box::new` in const context.
            return;
        }
        if !expected.is_box() || found.is_box() {
            return;
        }
        let boxed_found = self.tcx.mk_box(found);
        if let (true, Ok(snippet)) = (
            self.can_coerce(boxed_found, expected),
            self.sess().source_map().span_to_snippet(expr.span),
        ) {
            err.span_suggestion(
                expr.span,
                "store this in the heap by calling `Box::new`",
                format!("Box::new({})", snippet),
                Applicability::MachineApplicable,
            );
            err.note(
                "for more on the distinction between the stack and the heap, read \
                 https://doc.rust-lang.org/book/ch15-01-box.html, \
                 https://doc.rust-lang.org/rust-by-example/std/box.html, and \
                 https://doc.rust-lang.org/std/boxed/index.html",
            );
        }
    }

    /// When encountering an `impl Future` where `BoxFuture` is expected, suggest `Box::pin`.
    pub(in super::super) fn suggest_calling_boxed_future_when_appropriate(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        // Handle #68197.

        if self.tcx.hir().is_inside_const_context(expr.hir_id) {
            // Do not suggest `Box::new` in const context.
            return false;
        }
        let pin_did = self.tcx.lang_items().pin_type();
        match expected.kind() {
            ty::Adt(def, _) if Some(def.did) != pin_did => return false,
            // This guards the `unwrap` and `mk_box` below.
            _ if pin_did.is_none() || self.tcx.lang_items().owned_box().is_none() => return false,
            _ => {}
        }
        let boxed_found = self.tcx.mk_box(found);
        let new_found = self.tcx.mk_lang_item(boxed_found, LangItem::Pin).unwrap();
        if let (true, Ok(snippet)) = (
            self.can_coerce(new_found, expected),
            self.sess().source_map().span_to_snippet(expr.span),
        ) {
            match found.kind() {
                ty::Adt(def, _) if def.is_box() => {
                    err.help("use `Box::pin`");
                }
                _ => {
                    err.span_suggestion(
                        expr.span,
                        "you need to pin and box this expression",
                        format!("Box::pin({})", snippet),
                        Applicability::MachineApplicable,
                    );
                }
            }
            true
        } else {
            false
        }
    }

    /// A common error is to forget to add a semicolon at the end of a block, e.g.,
    ///
    /// ```
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return expression in a block would make sense on its own as a
    /// statement and the return type has been left as default or has been specified as `()`. If so,
    /// it suggests adding a semicolon.
    fn suggest_missing_semicolon(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expression: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        cause_span: Span,
    ) {
        if expected.is_unit() {
            // `BlockTailExpression` only relevant if the tail expr would be
            // useful on its own.
            match expression.kind {
                ExprKind::Call(..)
                | ExprKind::MethodCall(..)
                | ExprKind::Loop(..)
                | ExprKind::Match(..)
                | ExprKind::Block(..) => {
                    err.span_suggestion(
                        cause_span.shrink_to_hi(),
                        "try adding a semicolon",
                        ";".to_string(),
                        Applicability::MachineApplicable,
                    );
                }
                _ => (),
            }
        }
    }

    /// A possible error is to forget to add a return type that is needed:
    ///
    /// ```
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return type is left as default, the method is not part of an
    /// `impl` block and that it isn't the `main` method. If so, it suggests setting the return
    /// type.
    pub(in super::super) fn suggest_missing_return_type(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        fn_decl: &hir::FnDecl<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        can_suggest: bool,
    ) -> bool {
        // Only suggest changing the return type for methods that
        // haven't set a return type at all (and aren't `fn main()` or an impl).
        match (&fn_decl.output, found.is_suggestable(), can_suggest, expected.is_unit()) {
            (&hir::FnRetTy::DefaultReturn(span), true, true, true) => {
                err.span_suggestion(
                    span,
                    "try adding a return type",
                    format!("-> {} ", self.resolve_vars_with_obligations(found)),
                    Applicability::MachineApplicable,
                );
                true
            }
            (&hir::FnRetTy::DefaultReturn(span), false, true, true) => {
                err.span_label(span, "possibly return type missing here?");
                true
            }
            (&hir::FnRetTy::DefaultReturn(span), _, false, true) => {
                // `fn main()` must return `()`, do not suggest changing return type
                err.span_label(span, "expected `()` because of default return type");
                true
            }
            // expectation was caused by something else, not the default return
            (&hir::FnRetTy::DefaultReturn(_), _, _, false) => false,
            (&hir::FnRetTy::Return(ref ty), _, _, _) => {
                // Only point to return type if the expected type is the return type, as if they
                // are not, the expectation must have been caused by something else.
                debug!("suggest_missing_return_type: return type {:?} node {:?}", ty, ty.kind);
                let sp = ty.span;
                let ty = AstConv::ast_ty_to_ty(self, ty);
                debug!("suggest_missing_return_type: return type {:?}", ty);
                debug!("suggest_missing_return_type: expected type {:?}", ty);
                if ty.kind() == expected.kind() {
                    err.span_label(sp, format!("expected `{}` because of return type", expected));
                    return true;
                }
                false
            }
        }
    }

    pub(in super::super) fn suggest_missing_parentheses(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
    ) {
        let sp = self.tcx.sess.source_map().start_point(expr.span);
        if let Some(sp) = self.tcx.sess.parse_sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            // `{ 42 } &&x` (#61475) or `{ 42 } && if x { 1 } else { 0 }`
            self.tcx.sess.parse_sess.expr_parentheses_needed(err, *sp, None);
        }
    }
}

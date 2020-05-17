use if_chain::if_chain;
use rustc_ast::ast;
use rustc_ast::visit::FnKind;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::BytePos;

use crate::utils::{in_macro, match_path_ast, snippet_opt, span_lint_and_sugg, span_lint_and_then};

declare_clippy_lint! {
    /// **What it does:** Checks for return statements at the end of a block.
    ///
    /// **Why is this bad?** Removing the `return` and semicolon will make the code
    /// more rusty.
    ///
    /// **Known problems:** If the computation returning the value borrows a local
    /// variable, removing the `return` may run afoul of the borrow checker.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo(x: usize) -> usize {
    ///     return x;
    /// }
    /// ```
    /// simplify to
    /// ```rust
    /// fn foo(x: usize) -> usize {
    ///     x
    /// }
    /// ```
    pub NEEDLESS_RETURN,
    style,
    "using a return statement like `return expr;` where an expression would suffice"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `let`-bindings, which are subsequently
    /// returned.
    ///
    /// **Why is this bad?** It is just extraneous code. Remove it to make your code
    /// more rusty.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo() -> String {
    ///     let x = String::new();
    ///     x
    /// }
    /// ```
    /// instead, use
    /// ```
    /// fn foo() -> String {
    ///     String::new()
    /// }
    /// ```
    pub LET_AND_RETURN,
    style,
    "creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block"
}

declare_clippy_lint! {
    /// **What it does:** Checks for unit (`()`) expressions that can be removed.
    ///
    /// **Why is this bad?** Such expressions add no value, but can make the code
    /// less readable. Depending on formatting they can make a `break` or `return`
    /// statement look like a function call.
    ///
    /// **Known problems:** The lint currently misses unit return types in types,
    /// e.g., the `F` in `fn generic_unit<F: Fn() -> ()>(f: F) { .. }`.
    ///
    /// **Example:**
    /// ```rust
    /// fn return_unit() -> () {
    ///     ()
    /// }
    /// ```
    pub UNUSED_UNIT,
    style,
    "needless unit expression"
}

#[derive(PartialEq, Eq, Copy, Clone)]
enum RetReplacement {
    Empty,
    Block,
}

declare_lint_pass!(Return => [NEEDLESS_RETURN, LET_AND_RETURN, UNUSED_UNIT]);

impl Return {
    // Check the final stmt or expr in a block for unnecessary return.
    fn check_block_return(&mut self, cx: &EarlyContext<'_>, block: &ast::Block) {
        if let Some(stmt) = block.stmts.last() {
            match stmt.kind {
                ast::StmtKind::Expr(ref expr) | ast::StmtKind::Semi(ref expr) => {
                    self.check_final_expr(cx, expr, Some(stmt.span), RetReplacement::Empty);
                },
                _ => (),
            }
        }
    }

    // Check a the final expression in a block if it's a return.
    fn check_final_expr(
        &mut self,
        cx: &EarlyContext<'_>,
        expr: &ast::Expr,
        span: Option<Span>,
        replacement: RetReplacement,
    ) {
        match expr.kind {
            // simple return is always "bad"
            ast::ExprKind::Ret(ref inner) => {
                // allow `#[cfg(a)] return a; #[cfg(b)] return b;`
                if !expr.attrs.iter().any(attr_is_cfg) {
                    Self::emit_return_lint(
                        cx,
                        span.expect("`else return` is not possible"),
                        inner.as_ref().map(|i| i.span),
                        replacement,
                    );
                }
            },
            // a whole block? check it!
            ast::ExprKind::Block(ref block, _) => {
                self.check_block_return(cx, block);
            },
            // an if/if let expr, check both exprs
            // note, if without else is going to be a type checking error anyways
            // (except for unit type functions) so we don't match it
            ast::ExprKind::If(_, ref ifblock, Some(ref elsexpr)) => {
                self.check_block_return(cx, ifblock);
                self.check_final_expr(cx, elsexpr, None, RetReplacement::Empty);
            },
            // a match expr, check all arms
            ast::ExprKind::Match(_, ref arms) => {
                for arm in arms {
                    self.check_final_expr(cx, &arm.body, Some(arm.body.span), RetReplacement::Block);
                }
            },
            _ => (),
        }
    }

    fn emit_return_lint(cx: &EarlyContext<'_>, ret_span: Span, inner_span: Option<Span>, replacement: RetReplacement) {
        match inner_span {
            Some(inner_span) => {
                if in_external_macro(cx.sess(), inner_span) || inner_span.from_expansion() {
                    return;
                }

                span_lint_and_then(cx, NEEDLESS_RETURN, ret_span, "unneeded `return` statement", |diag| {
                    if let Some(snippet) = snippet_opt(cx, inner_span) {
                        diag.span_suggestion(ret_span, "remove `return`", snippet, Applicability::MachineApplicable);
                    }
                })
            },
            None => match replacement {
                RetReplacement::Empty => {
                    span_lint_and_sugg(
                        cx,
                        NEEDLESS_RETURN,
                        ret_span,
                        "unneeded `return` statement",
                        "remove `return`",
                        String::new(),
                        Applicability::MachineApplicable,
                    );
                },
                RetReplacement::Block => {
                    span_lint_and_sugg(
                        cx,
                        NEEDLESS_RETURN,
                        ret_span,
                        "unneeded `return` statement",
                        "replace `return` with an empty block",
                        "{}".to_string(),
                        Applicability::MachineApplicable,
                    );
                },
            },
        }
    }

    // Check for "let x = EXPR; x"
    fn check_let_return(cx: &EarlyContext<'_>, block: &ast::Block) {
        let mut it = block.stmts.iter();

        // we need both a let-binding stmt and an expr
        if_chain! {
            if let Some(retexpr) = it.next_back();
            if let ast::StmtKind::Expr(ref retexpr) = retexpr.kind;
            if let Some(stmt) = it.next_back();
            if let ast::StmtKind::Local(ref local) = stmt.kind;
            // don't lint in the presence of type inference
            if local.ty.is_none();
            if local.attrs.is_empty();
            if let Some(ref initexpr) = local.init;
            if let ast::PatKind::Ident(_, ident, _) = local.pat.kind;
            if let ast::ExprKind::Path(_, ref path) = retexpr.kind;
            if match_path_ast(path, &[&*ident.name.as_str()]);
            if !in_external_macro(cx.sess(), initexpr.span);
            if !in_external_macro(cx.sess(), retexpr.span);
            if !in_external_macro(cx.sess(), local.span);
            if !in_macro(local.span);
            then {
                span_lint_and_then(
                    cx,
                    LET_AND_RETURN,
                    retexpr.span,
                    "returning the result of a `let` binding from a block",
                    |err| {
                        err.span_label(local.span, "unnecessary `let` binding");

                        if let Some(snippet) = snippet_opt(cx, initexpr.span) {
                            err.multipart_suggestion(
                                "return the expression directly",
                                vec![
                                    (local.span, String::new()),
                                    (retexpr.span, snippet),
                                ],
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_help(initexpr.span, "this expression can be directly returned");
                        }
                    },
                );
            }
        }
    }
}

impl EarlyLintPass for Return {
    fn check_fn(&mut self, cx: &EarlyContext<'_>, kind: FnKind<'_>, span: Span, _: ast::NodeId) {
        match kind {
            FnKind::Fn(.., Some(block)) => self.check_block_return(cx, block),
            FnKind::Closure(_, body) => self.check_final_expr(cx, body, Some(body.span), RetReplacement::Empty),
            FnKind::Fn(.., None) => {},
        }
        if_chain! {
            if let ast::FnRetTy::Ty(ref ty) = kind.decl().output;
            if let ast::TyKind::Tup(ref vals) = ty.kind;
            if vals.is_empty() && !ty.span.from_expansion() && get_def(span) == get_def(ty.span);
            then {
                lint_unneeded_unit_return(cx, ty, span);
            }
        }
    }

    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &ast::Block) {
        Self::check_let_return(cx, block);
        if_chain! {
            if let Some(ref stmt) = block.stmts.last();
            if let ast::StmtKind::Expr(ref expr) = stmt.kind;
            if is_unit_expr(expr) && !stmt.span.from_expansion();
            then {
                let sp = expr.span;
                span_lint_and_sugg(
                    cx,
                    UNUSED_UNIT,
                    sp,
                    "unneeded unit expression",
                    "remove the final `()`",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        match e.kind {
            ast::ExprKind::Ret(Some(ref expr)) | ast::ExprKind::Break(_, Some(ref expr)) => {
                if is_unit_expr(expr) && !expr.span.from_expansion() {
                    span_lint_and_sugg(
                        cx,
                        UNUSED_UNIT,
                        expr.span,
                        "unneeded `()`",
                        "remove the `()`",
                        String::new(),
                        Applicability::MachineApplicable,
                    );
                }
            },
            _ => (),
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &EarlyContext<'_>, poly: &ast::PolyTraitRef, _: &ast::TraitBoundModifier) {
        let segments = &poly.trait_ref.path.segments;

        if_chain! {
            if segments.len() == 1;
            if ["Fn", "FnMut", "FnOnce"].contains(&&*segments[0].ident.name.as_str());
            if let Some(args) = &segments[0].args;
            if let ast::GenericArgs::Parenthesized(generic_args) = &**args;
            if let ast::FnRetTy::Ty(ty) = &generic_args.output;
            if ty.kind.is_unit();
            then {
                lint_unneeded_unit_return(cx, ty, generic_args.span);
            }
        }
    }
}

fn attr_is_cfg(attr: &ast::Attribute) -> bool {
    attr.meta_item_list().is_some() && attr.check_name(sym!(cfg))
}

// get the def site
#[must_use]
fn get_def(span: Span) -> Option<Span> {
    if span.from_expansion() {
        Some(span.ctxt().outer_expn_data().def_site)
    } else {
        None
    }
}

// is this expr a `()` unit?
fn is_unit_expr(expr: &ast::Expr) -> bool {
    if let ast::ExprKind::Tup(ref vals) = expr.kind {
        vals.is_empty()
    } else {
        false
    }
}

fn lint_unneeded_unit_return(cx: &EarlyContext<'_>, ty: &ast::Ty, span: Span) {
    let (ret_span, appl) = if let Ok(fn_source) = cx.sess().source_map().span_to_snippet(span.with_hi(ty.span.hi())) {
        if let Some(rpos) = fn_source.rfind("->") {
            #[allow(clippy::cast_possible_truncation)]
            (
                ty.span.with_lo(BytePos(span.lo().0 + rpos as u32)),
                Applicability::MachineApplicable,
            )
        } else {
            (ty.span, Applicability::MaybeIncorrect)
        }
    } else {
        (ty.span, Applicability::MaybeIncorrect)
    };
    span_lint_and_sugg(
        cx,
        UNUSED_UNIT,
        ret_span,
        "unneeded unit return type",
        "remove the `-> ()`",
        String::new(),
        appl,
    );
}

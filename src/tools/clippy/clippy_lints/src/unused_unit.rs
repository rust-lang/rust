use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{position_before_rarrow, snippet_opt};
use if_chain::if_chain;
use rustc_ast::{ast, visit::FnKind, ClosureBinder};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::BytePos;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unit (`()`) expressions that can be removed.
    ///
    /// ### Why is this bad?
    /// Such expressions add no value, but can make the code
    /// less readable. Depending on formatting they can make a `break` or `return`
    /// statement look like a function call.
    ///
    /// ### Example
    /// ```rust
    /// fn return_unit() -> () {
    ///     ()
    /// }
    /// ```
    /// is equivalent to
    /// ```rust
    /// fn return_unit() {}
    /// ```
    #[clippy::version = "1.31.0"]
    pub UNUSED_UNIT,
    style,
    "needless unit expression"
}

declare_lint_pass!(UnusedUnit => [UNUSED_UNIT]);

impl EarlyLintPass for UnusedUnit {
    fn check_fn(&mut self, cx: &EarlyContext<'_>, kind: FnKind<'_>, span: Span, _: ast::NodeId) {
        if_chain! {
            if let ast::FnRetTy::Ty(ref ty) = kind.decl().output;
            if let ast::TyKind::Tup(ref vals) = ty.kind;
            if vals.is_empty() && !ty.span.from_expansion() && get_def(span) == get_def(ty.span);
            then {
                // implicit types in closure signatures are forbidden when `for<...>` is present
                if let FnKind::Closure(&ClosureBinder::For { .. }, ..) = kind {
                    return;
                }

                lint_unneeded_unit_return(cx, ty, span);
            }
        }
    }

    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &ast::Block) {
        if_chain! {
            if let Some(stmt) = block.stmts.last();
            if let ast::StmtKind::Expr(ref expr) = stmt.kind;
            if is_unit_expr(expr);
            let ctxt = block.span.ctxt();
            if stmt.span.ctxt() == ctxt && expr.span.ctxt() == ctxt;
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

    fn check_poly_trait_ref(&mut self, cx: &EarlyContext<'_>, poly: &ast::PolyTraitRef) {
        let segments = &poly.trait_ref.path.segments;

        if_chain! {
            if segments.len() == 1;
            if ["Fn", "FnMut", "FnOnce"].contains(&segments[0].ident.name.as_str());
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
    let (ret_span, appl) =
        snippet_opt(cx, span.with_hi(ty.span.hi())).map_or((ty.span, Applicability::MaybeIncorrect), |fn_source| {
            position_before_rarrow(&fn_source).map_or((ty.span, Applicability::MaybeIncorrect), |rpos| {
                (
                    #[expect(clippy::cast_possible_truncation)]
                    ty.span.with_lo(BytePos(span.lo().0 + rpos as u32)),
                    Applicability::MachineApplicable,
                )
            })
        });
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

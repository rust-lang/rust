use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::source::{snippet, snippet_opt};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    self as hir, def, BinOpKind, BindingAnnotation, Body, ByRef, Expr, ExprKind, FnDecl, HirId, Mutability, PatKind,
    Stmt, StmtKind, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::hygiene::DesugaringKind;
use rustc_span::source_map::{ExpnKind, Span};

use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, in_constant, iter_input_pats, last_path_segment, SpanlessEq};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for function arguments and let bindings denoted as
    /// `ref`.
    ///
    /// ### Why is this bad?
    /// The `ref` declaration makes the function take an owned
    /// value, but turns the argument into a reference (which means that the value
    /// is destroyed when exiting the function). This adds not much value: either
    /// take a reference type, or take an owned value and create references in the
    /// body.
    ///
    /// For let bindings, `let x = &foo;` is preferred over `let ref x = foo`. The
    /// type of `x` is more obvious with the former.
    ///
    /// ### Known problems
    /// If the argument is dereferenced within the function,
    /// removing the `ref` will lead to errors. This can be fixed by removing the
    /// dereferences, e.g., changing `*x` to `x` within the function.
    ///
    /// ### Example
    /// ```rust
    /// fn foo(ref _x: u8) {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// fn foo(_x: &u8) {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TOPLEVEL_REF_ARG,
    style,
    "an entire binding declared as `ref`, in a function argument or a `let` statement"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of bindings with a single leading
    /// underscore.
    ///
    /// ### Why is this bad?
    /// A single leading underscore is usually used to indicate
    /// that a binding will not be used. Using such a binding breaks this
    /// expectation.
    ///
    /// ### Known problems
    /// The lint does not work properly with desugaring and
    /// macro, it has been allowed in the mean time.
    ///
    /// ### Example
    /// ```rust
    /// let _x = 0;
    /// let y = _x + 1; // Here we are using `_x`, even though it has a leading
    ///                 // underscore. We should rename `_x` to `x`
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USED_UNDERSCORE_BINDING,
    pedantic,
    "using a binding which is prefixed with an underscore"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of short circuit boolean conditions as
    /// a
    /// statement.
    ///
    /// ### Why is this bad?
    /// Using a short circuit boolean condition as a statement
    /// may hide the fact that the second part is executed or not depending on the
    /// outcome of the first part.
    ///
    /// ### Example
    /// ```rust,ignore
    /// f() && g(); // We should write `if f() { g(); }`.
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SHORT_CIRCUIT_STATEMENT,
    complexity,
    "using a short circuit boolean condition as a statement"
}

declare_clippy_lint! {
    /// ### What it does
    /// Catch casts from `0` to some pointer type
    ///
    /// ### Why is this bad?
    /// This generally means `null` and is better expressed as
    /// {`std`, `core`}`::ptr::`{`null`, `null_mut`}.
    ///
    /// ### Example
    /// ```rust
    /// let a = 0 as *const u32;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let a = std::ptr::null::<u32>();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ZERO_PTR,
    style,
    "using `0 as *{const, mut} T`"
}

declare_lint_pass!(MiscLints => [
    TOPLEVEL_REF_ARG,
    USED_UNDERSCORE_BINDING,
    SHORT_CIRCUIT_STATEMENT,
    ZERO_PTR,
]);

impl<'tcx> LateLintPass<'tcx> for MiscLints {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        k: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        _: HirId,
    ) {
        if let FnKind::Closure = k {
            // Does not apply to closures
            return;
        }
        if in_external_macro(cx.tcx.sess, span) {
            return;
        }
        for arg in iter_input_pats(decl, body) {
            if let PatKind::Binding(BindingAnnotation(ByRef::Yes, _), ..) = arg.pat.kind {
                span_lint(
                    cx,
                    TOPLEVEL_REF_ARG,
                    arg.pat.span,
                    "`ref` directly on a function argument is ignored. \
                    Consider using a reference type instead",
                );
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if_chain! {
            if !in_external_macro(cx.tcx.sess, stmt.span);
            if let StmtKind::Local(local) = stmt.kind;
            if let PatKind::Binding(BindingAnnotation(ByRef::Yes, mutabl), .., name, None) = local.pat.kind;
            if let Some(init) = local.init;
            then {
                // use the macro callsite when the init span (but not the whole local span)
                // comes from an expansion like `vec![1, 2, 3]` in `let ref _ = vec![1, 2, 3];`
                let sugg_init = if init.span.from_expansion() && !local.span.from_expansion() {
                    Sugg::hir_with_macro_callsite(cx, init, "..")
                } else {
                    Sugg::hir(cx, init, "..")
                };
                let (mutopt, initref) = if mutabl == Mutability::Mut {
                    ("mut ", sugg_init.mut_addr())
                } else {
                    ("", sugg_init.addr())
                };
                let tyopt = if let Some(ty) = local.ty {
                    format!(": &{mutopt}{ty}", mutopt=mutopt, ty=snippet(cx, ty.span, ".."))
                } else {
                    String::new()
                };
                span_lint_hir_and_then(
                    cx,
                    TOPLEVEL_REF_ARG,
                    init.hir_id,
                    local.pat.span,
                    "`ref` on an entire `let` pattern is discouraged, take a reference with `&` instead",
                    |diag| {
                        diag.span_suggestion(
                            stmt.span,
                            "try",
                            format!(
                                "let {name}{tyopt} = {initref};",
                                name=snippet(cx, name.span, ".."),
                                tyopt=tyopt,
                                initref=initref,
                            ),
                            Applicability::MachineApplicable,
                        );
                    }
                );
            }
        };
        if_chain! {
            if let StmtKind::Semi(expr) = stmt.kind;
            if let ExprKind::Binary(ref binop, a, b) = expr.kind;
            if binop.node == BinOpKind::And || binop.node == BinOpKind::Or;
            if let Some(sugg) = Sugg::hir_opt(cx, a);
            then {
                span_lint_hir_and_then(
                    cx,
                    SHORT_CIRCUIT_STATEMENT,
                    expr.hir_id,
                    stmt.span,
                    "boolean short circuit operator in statement may be clearer using an explicit test",
                    |diag| {
                        let sugg = if binop.node == BinOpKind::Or { !sugg } else { sugg };
                        diag.span_suggestion(
                            stmt.span,
                            "replace it with",
                            format!(
                                "if {} {{ {}; }}",
                                sugg,
                                &snippet(cx, b.span, ".."),
                            ),
                            Applicability::MachineApplicable, // snippet
                        );
                    });
            }
        };
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Cast(e, ty) = expr.kind {
            check_cast(cx, expr.span, e, ty);
            return;
        }
        if in_attributes_expansion(expr) || expr.span.is_desugaring(DesugaringKind::Await) {
            // Don't lint things expanded by #[derive(...)], etc or `await` desugaring
            return;
        }
        let sym;
        let binding = match expr.kind {
            ExprKind::Path(ref qpath) if !matches!(qpath, hir::QPath::LangItem(..)) => {
                let binding = last_path_segment(qpath).ident.as_str();
                if binding.starts_with('_') &&
                    !binding.starts_with("__") &&
                    binding != "_result" && // FIXME: #944
                    is_used(cx, expr) &&
                    // don't lint if the declaration is in a macro
                    non_macro_local(cx, cx.qpath_res(qpath, expr.hir_id))
                {
                    Some(binding)
                } else {
                    None
                }
            },
            ExprKind::Field(_, ident) => {
                sym = ident.name;
                let name = sym.as_str();
                if name.starts_with('_') && !name.starts_with("__") {
                    Some(name)
                } else {
                    None
                }
            },
            _ => None,
        };
        if let Some(binding) = binding {
            span_lint(
                cx,
                USED_UNDERSCORE_BINDING,
                expr.span,
                &format!(
                    "used binding `{}` which is prefixed with an underscore. A leading \
                     underscore signals that a binding will not be used",
                    binding
                ),
            );
        }
    }
}

/// Heuristic to see if an expression is used. Should be compatible with
/// `unused_variables`'s idea
/// of what it means for an expression to be "used".
fn is_used(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    get_parent_expr(cx, expr).map_or(true, |parent| match parent.kind {
        ExprKind::Assign(_, rhs, _) | ExprKind::AssignOp(_, _, rhs) => SpanlessEq::new(cx).eq_expr(rhs, expr),
        _ => is_used(cx, parent),
    })
}

/// Tests whether an expression is in a macro expansion (e.g., something
/// generated by `#[derive(...)]` or the like).
fn in_attributes_expansion(expr: &Expr<'_>) -> bool {
    use rustc_span::hygiene::MacroKind;
    if expr.span.from_expansion() {
        let data = expr.span.ctxt().outer_expn_data();
        matches!(data.kind, ExpnKind::Macro(MacroKind::Attr | MacroKind::Derive, _))
    } else {
        false
    }
}

/// Tests whether `res` is a variable defined outside a macro.
fn non_macro_local(cx: &LateContext<'_>, res: def::Res) -> bool {
    if let def::Res::Local(id) = res {
        !cx.tcx.hir().span(id).from_expansion()
    } else {
        false
    }
}

fn check_cast(cx: &LateContext<'_>, span: Span, e: &Expr<'_>, ty: &hir::Ty<'_>) {
    if_chain! {
        if let TyKind::Ptr(ref mut_ty) = ty.kind;
        if let ExprKind::Lit(ref lit) = e.kind;
        if let LitKind::Int(0, _) = lit.node;
        if !in_constant(cx, e.hir_id);
        then {
            let (msg, sugg_fn) = match mut_ty.mutbl {
                Mutability::Mut => ("`0 as *mut _` detected", "std::ptr::null_mut"),
                Mutability::Not => ("`0 as *const _` detected", "std::ptr::null"),
            };

            let (sugg, appl) = if let TyKind::Infer = mut_ty.ty.kind {
                (format!("{}()", sugg_fn), Applicability::MachineApplicable)
            } else if let Some(mut_ty_snip) = snippet_opt(cx, mut_ty.ty.span) {
                (format!("{}::<{}>()", sugg_fn, mut_ty_snip), Applicability::MachineApplicable)
            } else {
                // `MaybeIncorrect` as type inference may not work with the suggested code
                (format!("{}()", sugg_fn), Applicability::MaybeIncorrect)
            };
            span_lint_and_sugg(cx, ZERO_PTR, span, msg, "try", sugg, appl);
        }
    }
}

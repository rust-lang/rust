use crate::utils::{contains_name, higher, iter_input_pats, snippet, span_lint_and_then};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    Block, Body, Expr, ExprKind, FnDecl, Guard, HirId, Local, MutTy, Pat, PatKind, Path, QPath, StmtKind, Ty, TyKind,
    UnOp,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::symbol::Symbol;

declare_clippy_lint! {
    /// **What it does:** Checks for bindings that shadow other bindings already in
    /// scope, while just changing reference level or mutability.
    ///
    /// **Why is this bad?** Not much, in fact it's a very common pattern in Rust
    /// code. Still, some may opt to avoid it in their code base, they can set this
    /// lint to `Warn`.
    ///
    /// **Known problems:** This lint, as the other shadowing related lints,
    /// currently only catches very simple patterns.
    ///
    /// **Example:**
    /// ```rust
    /// # let x = 1;
    /// // Bad
    /// let x = &x;
    ///
    /// // Good
    /// let y = &x; // use different variable name
    /// ```
    pub SHADOW_SAME,
    restriction,
    "rebinding a name to itself, e.g., `let mut x = &mut x`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for bindings that shadow other bindings already in
    /// scope, while reusing the original value.
    ///
    /// **Why is this bad?** Not too much, in fact it's a common pattern in Rust
    /// code. Still, some argue that name shadowing like this hurts readability,
    /// because a value may be bound to different things depending on position in
    /// the code.
    ///
    /// **Known problems:** This lint, as the other shadowing related lints,
    /// currently only catches very simple patterns.
    ///
    /// **Example:**
    /// ```rust
    /// let x = 2;
    /// let x = x + 1;
    /// ```
    /// use different variable name:
    /// ```rust
    /// let x = 2;
    /// let y = x + 1;
    /// ```
    pub SHADOW_REUSE,
    restriction,
    "rebinding a name to an expression that re-uses the original value, e.g., `let x = x + 1`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for bindings that shadow other bindings already in
    /// scope, either without a initialization or with one that does not even use
    /// the original value.
    ///
    /// **Why is this bad?** Name shadowing can hurt readability, especially in
    /// large code bases, because it is easy to lose track of the active binding at
    /// any place in the code. This can be alleviated by either giving more specific
    /// names to bindings or introducing more scopes to contain the bindings.
    ///
    /// **Known problems:** This lint, as the other shadowing related lints,
    /// currently only catches very simple patterns. Note that
    /// `allow`/`warn`/`deny`/`forbid` attributes only work on the function level
    /// for this lint.
    ///
    /// **Example:**
    /// ```rust
    /// # let y = 1;
    /// # let z = 2;
    /// let x = y;
    ///
    /// // Bad
    /// let x = z; // shadows the earlier binding
    ///
    /// // Good
    /// let w = z; // use different variable name
    /// ```
    pub SHADOW_UNRELATED,
    pedantic,
    "rebinding a name without even using the original value"
}

declare_lint_pass!(Shadow => [SHADOW_SAME, SHADOW_REUSE, SHADOW_UNRELATED]);

impl<'tcx> LateLintPass<'tcx> for Shadow {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        _: HirId,
    ) {
        if in_external_macro(cx.sess(), body.value.span) {
            return;
        }
        check_fn(cx, decl, body);
    }
}

fn check_fn<'tcx>(cx: &LateContext<'tcx>, decl: &'tcx FnDecl<'_>, body: &'tcx Body<'_>) {
    let mut bindings = Vec::with_capacity(decl.inputs.len());
    for arg in iter_input_pats(decl, body) {
        if let PatKind::Binding(.., ident, _) = arg.pat.kind {
            bindings.push((ident.name, ident.span))
        }
    }
    check_expr(cx, &body.value, &mut bindings);
}

fn check_block<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'_>, bindings: &mut Vec<(Symbol, Span)>) {
    let len = bindings.len();
    for stmt in block.stmts {
        match stmt.kind {
            StmtKind::Local(ref local) => check_local(cx, local, bindings),
            StmtKind::Expr(ref e) | StmtKind::Semi(ref e) => check_expr(cx, e, bindings),
            StmtKind::Item(..) => {},
        }
    }
    if let Some(ref o) = block.expr {
        check_expr(cx, o, bindings);
    }
    bindings.truncate(len);
}

fn check_local<'tcx>(cx: &LateContext<'tcx>, local: &'tcx Local<'_>, bindings: &mut Vec<(Symbol, Span)>) {
    if in_external_macro(cx.sess(), local.span) {
        return;
    }
    if higher::is_from_for_desugar(local) {
        return;
    }
    let Local {
        ref pat,
        ref ty,
        ref init,
        span,
        ..
    } = *local;
    if let Some(ref t) = *ty {
        check_ty(cx, t, bindings)
    }
    if let Some(ref o) = *init {
        check_expr(cx, o, bindings);
        check_pat(cx, pat, Some(o), span, bindings);
    } else {
        check_pat(cx, pat, None, span, bindings);
    }
}

fn is_binding(cx: &LateContext<'_>, pat_id: HirId) -> bool {
    let var_ty = cx.typeck_results().node_type_opt(pat_id);
    var_ty.map_or(false, |var_ty| !matches!(var_ty.kind(), ty::Adt(..)))
}

fn check_pat<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    init: Option<&'tcx Expr<'_>>,
    span: Span,
    bindings: &mut Vec<(Symbol, Span)>,
) {
    // TODO: match more stuff / destructuring
    match pat.kind {
        PatKind::Binding(.., ident, ref inner) => {
            let name = ident.name;
            if is_binding(cx, pat.hir_id) {
                let mut new_binding = true;
                for tup in bindings.iter_mut() {
                    if tup.0 == name {
                        lint_shadow(cx, name, span, pat.span, init, tup.1);
                        tup.1 = ident.span;
                        new_binding = false;
                        break;
                    }
                }
                if new_binding {
                    bindings.push((name, ident.span));
                }
            }
            if let Some(ref p) = *inner {
                check_pat(cx, p, init, span, bindings);
            }
        },
        PatKind::Struct(_, pfields, _) => {
            if let Some(init_struct) = init {
                if let ExprKind::Struct(_, ref efields, _) = init_struct.kind {
                    for field in pfields {
                        let name = field.ident.name;
                        let efield = efields
                            .iter()
                            .find_map(|f| if f.ident.name == name { Some(&*f.expr) } else { None });
                        check_pat(cx, &field.pat, efield, span, bindings);
                    }
                } else {
                    for field in pfields {
                        check_pat(cx, &field.pat, init, span, bindings);
                    }
                }
            } else {
                for field in pfields {
                    check_pat(cx, &field.pat, None, span, bindings);
                }
            }
        },
        PatKind::Tuple(inner, _) => {
            if let Some(init_tup) = init {
                if let ExprKind::Tup(ref tup) = init_tup.kind {
                    for (i, p) in inner.iter().enumerate() {
                        check_pat(cx, p, Some(&tup[i]), p.span, bindings);
                    }
                } else {
                    for p in inner {
                        check_pat(cx, p, init, span, bindings);
                    }
                }
            } else {
                for p in inner {
                    check_pat(cx, p, None, span, bindings);
                }
            }
        },
        PatKind::Box(ref inner) => {
            if let Some(initp) = init {
                if let ExprKind::Box(ref inner_init) = initp.kind {
                    check_pat(cx, inner, Some(&**inner_init), span, bindings);
                } else {
                    check_pat(cx, inner, init, span, bindings);
                }
            } else {
                check_pat(cx, inner, init, span, bindings);
            }
        },
        PatKind::Ref(ref inner, _) => check_pat(cx, inner, init, span, bindings),
        // PatVec(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>),
        _ => (),
    }
}

fn lint_shadow<'tcx>(
    cx: &LateContext<'tcx>,
    name: Symbol,
    span: Span,
    pattern_span: Span,
    init: Option<&'tcx Expr<'_>>,
    prev_span: Span,
) {
    if let Some(expr) = init {
        if is_self_shadow(name, expr) {
            span_lint_and_then(
                cx,
                SHADOW_SAME,
                span,
                &format!(
                    "`{}` is shadowed by itself in `{}`",
                    snippet(cx, pattern_span, "_"),
                    snippet(cx, expr.span, "..")
                ),
                |diag| {
                    diag.span_note(prev_span, "previous binding is here");
                },
            );
        } else if contains_name(name, expr) {
            span_lint_and_then(
                cx,
                SHADOW_REUSE,
                pattern_span,
                &format!(
                    "`{}` is shadowed by `{}` which reuses the original value",
                    snippet(cx, pattern_span, "_"),
                    snippet(cx, expr.span, "..")
                ),
                |diag| {
                    diag.span_note(expr.span, "initialization happens here");
                    diag.span_note(prev_span, "previous binding is here");
                },
            );
        } else {
            span_lint_and_then(
                cx,
                SHADOW_UNRELATED,
                pattern_span,
                &format!("`{}` is being shadowed", snippet(cx, pattern_span, "_")),
                |diag| {
                    diag.span_note(expr.span, "initialization happens here");
                    diag.span_note(prev_span, "previous binding is here");
                },
            );
        }
    } else {
        span_lint_and_then(
            cx,
            SHADOW_UNRELATED,
            span,
            &format!("`{}` shadows a previous declaration", snippet(cx, pattern_span, "_")),
            |diag| {
                diag.span_note(prev_span, "previous binding is here");
            },
        );
    }
}

fn check_expr<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, bindings: &mut Vec<(Symbol, Span)>) {
    if in_external_macro(cx.sess(), expr.span) {
        return;
    }
    match expr.kind {
        ExprKind::Unary(_, ref e)
        | ExprKind::Field(ref e, _)
        | ExprKind::AddrOf(_, _, ref e)
        | ExprKind::Box(ref e) => check_expr(cx, e, bindings),
        ExprKind::Block(ref block, _) | ExprKind::Loop(ref block, _, _) => check_block(cx, block, bindings),
        // ExprKind::Call
        // ExprKind::MethodCall
        ExprKind::Array(v) | ExprKind::Tup(v) => {
            for e in v {
                check_expr(cx, e, bindings)
            }
        },
        ExprKind::Match(ref init, arms, _) => {
            check_expr(cx, init, bindings);
            let len = bindings.len();
            for arm in arms {
                check_pat(cx, &arm.pat, Some(&**init), arm.pat.span, bindings);
                // This is ugly, but needed to get the right type
                if let Some(ref guard) = arm.guard {
                    match guard {
                        Guard::If(if_expr) => check_expr(cx, if_expr, bindings),
                        Guard::IfLet(guard_pat, guard_expr) => {
                            check_pat(cx, guard_pat, Some(*guard_expr), guard_pat.span, bindings);
                            check_expr(cx, guard_expr, bindings);
                        },
                    }
                }
                check_expr(cx, &arm.body, bindings);
                bindings.truncate(len);
            }
        },
        _ => (),
    }
}

fn check_ty<'tcx>(cx: &LateContext<'tcx>, ty: &'tcx Ty<'_>, bindings: &mut Vec<(Symbol, Span)>) {
    match ty.kind {
        TyKind::Slice(ref sty) => check_ty(cx, sty, bindings),
        TyKind::Array(ref fty, ref anon_const) => {
            check_ty(cx, fty, bindings);
            check_expr(cx, &cx.tcx.hir().body(anon_const.body).value, bindings);
        },
        TyKind::Ptr(MutTy { ty: ref mty, .. }) | TyKind::Rptr(_, MutTy { ty: ref mty, .. }) => {
            check_ty(cx, mty, bindings)
        },
        TyKind::Tup(tup) => {
            for t in tup {
                check_ty(cx, t, bindings)
            }
        },
        TyKind::Typeof(ref anon_const) => check_expr(cx, &cx.tcx.hir().body(anon_const.body).value, bindings),
        _ => (),
    }
}

fn is_self_shadow(name: Symbol, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Box(ref inner) | ExprKind::AddrOf(_, _, ref inner) => is_self_shadow(name, inner),
        ExprKind::Block(ref block, _) => {
            block.stmts.is_empty() && block.expr.as_ref().map_or(false, |e| is_self_shadow(name, e))
        },
        ExprKind::Unary(op, ref inner) => (UnOp::UnDeref == op) && is_self_shadow(name, inner),
        ExprKind::Path(QPath::Resolved(_, ref path)) => path_eq_name(name, path),
        _ => false,
    }
}

fn path_eq_name(name: Symbol, path: &Path<'_>) -> bool {
    !path.is_global() && path.segments.len() == 1 && path.segments[0].ident.as_str() == name.as_str()
}

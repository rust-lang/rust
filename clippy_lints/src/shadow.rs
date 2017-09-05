use reexport::*;
use rustc::lint::*;
use rustc::hir::*;
use rustc::hir::intravisit::FnKind;
use rustc::ty;
use syntax::codemap::Span;
use utils::{contains_name, higher, in_external_macro, iter_input_pats, snippet, span_lint_and_then};

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
/// let x = &x;
/// ```
declare_lint! {
    pub SHADOW_SAME,
    Allow,
    "rebinding a name to itself, e.g. `let mut x = &mut x`"
}

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
/// let x = x + 1;
/// ```
declare_lint! {
    pub SHADOW_REUSE,
    Allow,
    "rebinding a name to an expression that re-uses the original value, e.g. \
     `let x = x + 1`"
}

/// **What it does:** Checks for bindings that shadow other bindings already in
/// scope, either without a initialization or with one that does not even use
/// the original value.
///
/// **Why is this bad?** Name shadowing can hurt readability, especially in
/// large code bases, because it is easy to lose track of the active binding at
/// any place in the code. This can be alleviated by either giving more specific
/// names to bindings ore introducing more scopes to contain the bindings.
///
/// **Known problems:** This lint, as the other shadowing related lints,
/// currently only catches very simple patterns.
///
/// **Example:**
/// ```rust
/// let x = y; let x = z; // shadows the earlier binding
/// ```
declare_lint! {
    pub SHADOW_UNRELATED,
    Allow,
    "rebinding a name without even using the original value"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SHADOW_SAME, SHADOW_REUSE, SHADOW_UNRELATED)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        _: Span,
        _: NodeId,
    ) {
        if in_external_macro(cx, body.value.span) {
            return;
        }
        check_fn(cx, decl, body);
    }
}

fn check_fn<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, decl: &'tcx FnDecl, body: &'tcx Body) {
    let mut bindings = Vec::new();
    for arg in iter_input_pats(decl, body) {
        if let PatKind::Binding(_, _, ident, _) = arg.pat.node {
            bindings.push((ident.node, ident.span))
        }
    }
    check_expr(cx, &body.value, &mut bindings);
}

fn check_block<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, block: &'tcx Block, bindings: &mut Vec<(Name, Span)>) {
    let len = bindings.len();
    for stmt in &block.stmts {
        match stmt.node {
            StmtDecl(ref decl, _) => check_decl(cx, decl, bindings),
            StmtExpr(ref e, _) | StmtSemi(ref e, _) => check_expr(cx, e, bindings),
        }
    }
    if let Some(ref o) = block.expr {
        check_expr(cx, o, bindings);
    }
    bindings.truncate(len);
}

fn check_decl<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, decl: &'tcx Decl, bindings: &mut Vec<(Name, Span)>) {
    if in_external_macro(cx, decl.span) {
        return;
    }
    if higher::is_from_for_desugar(decl) {
        return;
    }
    if let DeclLocal(ref local) = decl.node {
        let Local {
            ref pat,
            ref ty,
            ref init,
            span,
            ..
        } = **local;
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
}

fn is_binding(cx: &LateContext, pat_id: HirId) -> bool {
    let var_ty = cx.tables.node_id_to_type(pat_id);
    match var_ty.sty {
        ty::TyAdt(..) => false,
        _ => true,
    }
}

fn check_pat<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    pat: &'tcx Pat,
    init: Option<&'tcx Expr>,
    span: Span,
    bindings: &mut Vec<(Name, Span)>,
) {
    // TODO: match more stuff / destructuring
    match pat.node {
        PatKind::Binding(_, _, ref ident, ref inner) => {
            let name = ident.node;
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
        PatKind::Struct(_, ref pfields, _) => if let Some(init_struct) = init {
            if let ExprStruct(_, ref efields, _) = init_struct.node {
                for field in pfields {
                    let name = field.node.name;
                    let efield = efields
                        .iter()
                        .find(|f| f.name.node == name)
                        .map(|f| &*f.expr);
                    check_pat(cx, &field.node.pat, efield, span, bindings);
                }
            } else {
                for field in pfields {
                    check_pat(cx, &field.node.pat, init, span, bindings);
                }
            }
        } else {
            for field in pfields {
                check_pat(cx, &field.node.pat, None, span, bindings);
            }
        },
        PatKind::Tuple(ref inner, _) => if let Some(init_tup) = init {
            if let ExprTup(ref tup) = init_tup.node {
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
        },
        PatKind::Box(ref inner) => if let Some(initp) = init {
            if let ExprBox(ref inner_init) = initp.node {
                check_pat(cx, inner, Some(&**inner_init), span, bindings);
            } else {
                check_pat(cx, inner, init, span, bindings);
            }
        } else {
            check_pat(cx, inner, init, span, bindings);
        },
        PatKind::Ref(ref inner, _) => check_pat(cx, inner, init, span, bindings),
        // PatVec(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>),
        _ => (),
    }
}

fn lint_shadow<'a, 'tcx: 'a>(
    cx: &LateContext<'a, 'tcx>,
    name: Name,
    span: Span,
    pattern_span: Span,
    init: Option<&'tcx Expr>,
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
                |db| { db.span_note(prev_span, "previous binding is here"); },
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
                |db| {
                    db.span_note(expr.span, "initialization happens here");
                    db.span_note(prev_span, "previous binding is here");
                },
            );
        } else {
            span_lint_and_then(
                cx,
                SHADOW_UNRELATED,
                pattern_span,
                &format!(
                    "`{}` is shadowed by `{}`",
                    snippet(cx, pattern_span, "_"),
                    snippet(cx, expr.span, "..")
                ),
                |db| {
                    db.span_note(expr.span, "initialization happens here");
                    db.span_note(prev_span, "previous binding is here");
                },
            );
        }
    } else {
        span_lint_and_then(
            cx,
            SHADOW_UNRELATED,
            span,
            &format!("`{}` shadows a previous declaration", snippet(cx, pattern_span, "_")),
            |db| { db.span_note(prev_span, "previous binding is here"); },
        );
    }
}

fn check_expr<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr, bindings: &mut Vec<(Name, Span)>) {
    if in_external_macro(cx, expr.span) {
        return;
    }
    match expr.node {
        ExprUnary(_, ref e) | ExprField(ref e, _) | ExprTupField(ref e, _) | ExprAddrOf(_, ref e) | ExprBox(ref e) => {
            check_expr(cx, e, bindings)
        },
        ExprBlock(ref block) | ExprLoop(ref block, _, _) => check_block(cx, block, bindings),
        // ExprCall
        // ExprMethodCall
        ExprArray(ref v) | ExprTup(ref v) => for e in v {
            check_expr(cx, e, bindings)
        },
        ExprIf(ref cond, ref then, ref otherwise) => {
            check_expr(cx, cond, bindings);
            check_expr(cx, &**then, bindings);
            if let Some(ref o) = *otherwise {
                check_expr(cx, o, bindings);
            }
        },
        ExprWhile(ref cond, ref block, _) => {
            check_expr(cx, cond, bindings);
            check_block(cx, block, bindings);
        },
        ExprMatch(ref init, ref arms, _) => {
            check_expr(cx, init, bindings);
            let len = bindings.len();
            for arm in arms {
                for pat in &arm.pats {
                    check_pat(cx, pat, Some(&**init), pat.span, bindings);
                    // This is ugly, but needed to get the right type
                    if let Some(ref guard) = arm.guard {
                        check_expr(cx, guard, bindings);
                    }
                    check_expr(cx, &arm.body, bindings);
                    bindings.truncate(len);
                }
            }
        },
        _ => (),
    }
}

fn check_ty<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: &'tcx Ty, bindings: &mut Vec<(Name, Span)>) {
    match ty.node {
        TySlice(ref sty) => check_ty(cx, sty, bindings),
        TyArray(ref fty, body_id) => {
            check_ty(cx, fty, bindings);
            check_expr(cx, &cx.tcx.hir.body(body_id).value, bindings);
        },
        TyPtr(MutTy { ty: ref mty, .. }) | TyRptr(_, MutTy { ty: ref mty, .. }) => check_ty(cx, mty, bindings),
        TyTup(ref tup) => for t in tup {
            check_ty(cx, t, bindings)
        },
        TyTypeof(body_id) => check_expr(cx, &cx.tcx.hir.body(body_id).value, bindings),
        _ => (),
    }
}

fn is_self_shadow(name: Name, expr: &Expr) -> bool {
    match expr.node {
        ExprBox(ref inner) | ExprAddrOf(_, ref inner) => is_self_shadow(name, inner),
        ExprBlock(ref block) => {
            block.stmts.is_empty() &&
                block
                    .expr
                    .as_ref()
                    .map_or(false, |e| is_self_shadow(name, e))
        },
        ExprUnary(op, ref inner) => (UnDeref == op) && is_self_shadow(name, inner),
        ExprPath(QPath::Resolved(_, ref path)) => path_eq_name(name, path),
        _ => false,
    }
}

fn path_eq_name(name: Name, path: &Path) -> bool {
    !path.is_global() && path.segments.len() == 1 && path.segments[0].name.as_str() == name.as_str()
}

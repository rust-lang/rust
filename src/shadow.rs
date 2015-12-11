use std::ops::Deref;
use rustc_front::hir::*;
use reexport::*;
use syntax::codemap::Span;
use rustc_front::intravisit::{Visitor, FnKind};

use rustc::lint::*;
use rustc::middle::def::Def::{DefVariant, DefStruct};

use utils::{is_from_for_desugar, in_external_macro, snippet, span_lint, span_note_and_lint};

/// **What it does:** This lint checks for bindings that shadow other bindings already in scope, while just changing reference level or mutability. It is `Allow` by default.
///
/// **Why is this bad?** Not much, in fact it's a very common pattern in Rust code. Still, some may opt to avoid it in their code base, they can set this lint to `Warn`.
///
/// **Known problems:** This lint, as the other shadowing related lints, currently only catches very simple patterns.
///
/// **Example:** `let x = &x;`
declare_lint!(pub SHADOW_SAME, Allow,
    "rebinding a name to itself, e.g. `let mut x = &mut x`");
/// **What it does:** This lint checks for bindings that shadow other bindings already in scope, while reusing the original value. It is `Allow` by default.
///
/// **Why is this bad?** Not too much, in fact it's a common pattern in Rust code. Still, some argue that name shadowing like this hurts readability, because a value may be bound to different things depending on position in the code.
///
/// **Known problems:** This lint, as the other shadowing related lints, currently only catches very simple patterns.
///
/// **Example:** `let x = x + 1;`
declare_lint!(pub SHADOW_REUSE, Allow,
    "rebinding a name to an expression that re-uses the original value, e.g. \
    `let x = x + 1`");
/// **What it does:** This lint checks for bindings that shadow other bindings already in scope, either without a initialization or with one that does not even use the original value. This lint is `Warn` by default.
///
/// **Why is this bad?** Name shadowing can hurt readability, especially in large code bases, because it is easy to lose track of the active binding at any place in the code. This can be alleviated by either giving more specific names to bindings ore introducing more scopes to contain the bindings.
///
/// **Known problems:** This lint, as the other shadowing related lints, currently only catches very simple patterns.
///
/// **Example:** `let x = y; let x = z; // shadows the earlier binding`
declare_lint!(pub SHADOW_UNRELATED, Allow,
    "The name is re-bound without even using the original value");

#[derive(Copy, Clone)]
pub struct ShadowPass;

impl LintPass for ShadowPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SHADOW_SAME, SHADOW_REUSE, SHADOW_UNRELATED)
    }

}

impl LateLintPass for ShadowPass {
    fn check_fn(&mut self, cx: &LateContext, _: FnKind, decl: &FnDecl,
            block: &Block, _: Span, _: NodeId) {
        if in_external_macro(cx, block.span) { return; }
        check_fn(cx, decl, block);
    }
}

fn check_fn(cx: &LateContext, decl: &FnDecl, block: &Block) {
    let mut bindings = Vec::new();
    for arg in &decl.inputs {
        if let PatIdent(_, ident, _) = arg.pat.node {
            bindings.push((ident.node.unhygienic_name, ident.span))
        }
    }
    check_block(cx, block, &mut bindings);
}

fn check_block(cx: &LateContext, block: &Block, bindings: &mut Vec<(Name, Span)>) {
    let len = bindings.len();
    for stmt in &block.stmts {
        match stmt.node {
            StmtDecl(ref decl, _) => check_decl(cx, decl, bindings),
            StmtExpr(ref e, _) | StmtSemi(ref e, _) =>
                check_expr(cx, e, bindings)
        }
    }
    if let Some(ref o) = block.expr { check_expr(cx, o, bindings); }
    bindings.truncate(len);
}

fn check_decl(cx: &LateContext, decl: &Decl, bindings: &mut Vec<(Name, Span)>) {
    if in_external_macro(cx, decl.span) { return; }
    if is_from_for_desugar(decl) { return; }
    if let DeclLocal(ref local) = decl.node {
        let Local{ ref pat, ref ty, ref init, span, .. } = **local;
        if let Some(ref t) = *ty { check_ty(cx, t, bindings) }
        if let Some(ref o) = *init {
            check_expr(cx, o, bindings);
            check_pat(cx, pat, &Some(o), span, bindings);
        } else {
            check_pat(cx, pat, &None, span, bindings);
        }
    }
}

fn is_binding(cx: &LateContext, pat: &Pat) -> bool {
    match cx.tcx.def_map.borrow().get(&pat.id).map(|d| d.full_def()) {
        Some(DefVariant(..)) | Some(DefStruct(..)) => false,
        _ => true
    }
}

fn check_pat(cx: &LateContext, pat: &Pat, init: &Option<&Expr>, span: Span,
        bindings: &mut Vec<(Name, Span)>) {
    //TODO: match more stuff / destructuring
    match pat.node {
        PatIdent(_, ref ident, ref inner) => {
            let name = ident.node.unhygienic_name;
            if is_binding(cx, pat) {
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
            if let Some(ref p) = *inner { check_pat(cx, p, init, span, bindings); }
        }
        //PatEnum(Path, Option<Vec<P<Pat>>>),
        PatStruct(_, ref pfields, _) =>
            if let Some(ref init_struct) = *init {
                if let ExprStruct(_, ref efields, _) = init_struct.node {
                    for field in pfields {
                        let name = field.node.name;
                        let efield = efields.iter()
                            .find(|ref f| f.name.node == name)
                            .map(|f| &*f.expr);
                        check_pat(cx, &field.node.pat, &efield, span, bindings);
                    }
                } else {
                    for field in pfields {
                        check_pat(cx, &field.node.pat, init, span, bindings);
                    }
                }
            } else {
                for field in pfields {
                    check_pat(cx, &field.node.pat, &None, span, bindings);
                }
            },
        PatTup(ref inner) =>
            if let Some(ref init_tup) = *init {
                if let ExprTup(ref tup) = init_tup.node {
                    for (i, p) in inner.iter().enumerate() {
                        check_pat(cx, p, &Some(&tup[i]), p.span, bindings);
                    }
                } else {
                    for p in inner {
                        check_pat(cx, p, init, span, bindings);
                    }
                }
            } else {
                for p in inner {
                    check_pat(cx, p, &None, span, bindings);
                }
            },
        PatBox(ref inner) => {
            if let Some(ref initp) = *init {
                if let ExprBox(ref inner_init) = initp.node {
                    check_pat(cx, inner, &Some(&**inner_init), span, bindings);
                } else {
                    check_pat(cx, inner, init, span, bindings);
                }
            } else {
                check_pat(cx, inner, init, span, bindings);
            }
        }
        PatRegion(ref inner, _) =>
            check_pat(cx, inner, init, span, bindings),
        //PatVec(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>),
        _ => (),
    }
}

fn lint_shadow<T>(cx: &LateContext, name: Name, span: Span, lspan: Span, init:
        &Option<T>, prev_span: Span) where T: Deref<Target=Expr> {
    fn note_orig(cx: &LateContext, lint: &'static Lint, span: Span) {
        if cx.current_level(lint) != Level::Allow {
            cx.sess().span_note(span, "previous binding is here");
        }
    }
    if let Some(ref expr) = *init {
        if is_self_shadow(name, expr) {
            span_lint(cx, SHADOW_SAME, span, &format!(
                "{} is shadowed by itself in {}",
                snippet(cx, lspan, "_"),
                snippet(cx, expr.span, "..")));
                note_orig(cx, SHADOW_SAME, prev_span);
        } else {
            if contains_self(name, expr) {
                span_note_and_lint(cx, SHADOW_REUSE, lspan, &format!(
                    "{} is shadowed by {} which reuses the original value",
                    snippet(cx, lspan, "_"),
                    snippet(cx, expr.span, "..")),
                    expr.span, "initialization happens here");
                note_orig(cx, SHADOW_REUSE, prev_span);
            } else {
                span_note_and_lint(cx, SHADOW_UNRELATED, lspan, &format!(
                    "{} is shadowed by {}",
                    snippet(cx, lspan, "_"),
                    snippet(cx, expr.span, "..")),
                    expr.span, "initialization happens here");
                note_orig(cx, SHADOW_UNRELATED, prev_span);
            }
        }
    } else {
        span_lint(cx, SHADOW_UNRELATED, span, &format!(
            "{} shadows a previous declaration", snippet(cx, lspan, "_")));
        note_orig(cx, SHADOW_UNRELATED, prev_span);
    }
}

fn check_expr(cx: &LateContext, expr: &Expr, bindings: &mut Vec<(Name, Span)>) {
    if in_external_macro(cx, expr.span) { return; }
    match expr.node {
        ExprUnary(_, ref e) | ExprField(ref e, _) |
        ExprTupField(ref e, _) | ExprAddrOf(_, ref e) | ExprBox(ref e)
            => { check_expr(cx, e, bindings) }
        ExprBlock(ref block) | ExprLoop(ref block, _) =>
            { check_block(cx, block, bindings) }
        //ExprCall
        //ExprMethodCall
        ExprVec(ref v) | ExprTup(ref v) =>
            for ref e in v { check_expr(cx, e, bindings) },
        ExprIf(ref cond, ref then, ref otherwise) => {
            check_expr(cx, cond, bindings);
            check_block(cx, then, bindings);
            if let Some(ref o) = *otherwise { check_expr(cx, o, bindings); }
        }
        ExprWhile(ref cond, ref block, _) => {
            check_expr(cx, cond, bindings);
            check_block(cx, block, bindings);
        }
        ExprMatch(ref init, ref arms, _) => {
            check_expr(cx, init, bindings);
            let len = bindings.len();
            for ref arm in arms {
                for ref pat in &arm.pats {
                    check_pat(cx, &pat, &Some(&**init), pat.span, bindings);
                    //This is ugly, but needed to get the right type
                    if let Some(ref guard) = arm.guard {
                        check_expr(cx, guard, bindings);
                    }
                    check_expr(cx, &arm.body, bindings);
                    bindings.truncate(len);
                }
            }
        }
        _ => ()
    }
}

fn check_ty(cx: &LateContext, ty: &Ty, bindings: &mut Vec<(Name, Span)>) {
    match ty.node {
        TyObjectSum(ref sty, _) |
        TyVec(ref sty) => check_ty(cx, sty, bindings),
        TyFixedLengthVec(ref fty, ref expr) => {
            check_ty(cx, fty, bindings);
            check_expr(cx, expr, bindings);
        }
        TyPtr(MutTy{ ty: ref mty, .. }) |
        TyRptr(_, MutTy{ ty: ref mty, .. }) => check_ty(cx, mty, bindings),
        TyTup(ref tup) => { for ref t in tup { check_ty(cx, t, bindings) } }
        TyTypeof(ref expr) => check_expr(cx, expr, bindings),
        _ => (),
    }
}

fn is_self_shadow(name: Name, expr: &Expr) -> bool {
    match expr.node {
        ExprBox(ref inner) |
        ExprAddrOf(_, ref inner) => is_self_shadow(name, inner),
        ExprBlock(ref block) => block.stmts.is_empty() && block.expr.as_ref().
            map_or(false, |ref e| is_self_shadow(name, e)),
        ExprUnary(op, ref inner) => (UnDeref == op) &&
            is_self_shadow(name, inner),
        ExprPath(_, ref path) => path_eq_name(name, path),
        _ => false,
    }
}

fn path_eq_name(name: Name, path: &Path) -> bool {
    !path.global && path.segments.len() == 1 &&
        path.segments[0].identifier.unhygienic_name == name
}

struct ContainsSelf {
    name: Name,
    result: bool
}

impl<'v> Visitor<'v> for ContainsSelf {
    fn visit_ident(&mut self, _: Span, ident: Ident) {
        if self.name == ident.unhygienic_name {
            self.result = true;
        }
    }
}

fn contains_self(name: Name, expr: &Expr) -> bool {
    let mut cs = ContainsSelf { name: name, result: false };
    cs.visit_expr(expr);
    cs.result
}

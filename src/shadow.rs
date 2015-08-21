use syntax::ast::*;
use syntax::codemap::Span;
use syntax::visit::FnKind;

use rustc::lint::{Context, LintArray, LintPass};
use utils::{in_external_macro, snippet, span_lint};

declare_lint!(pub SHADOW_SAME, Allow,
    "rebinding a name to itself, e.g. `let mut x = &mut x`");
declare_lint!(pub SHADOW_REUSE, Allow,
    "rebinding a name to an expression that re-uses the original value, e.g. \
    `let x = x + 1`");
declare_lint!(pub SHADOW_FOREIGN, Warn,
    "The name is re-bound without even using the original value");

#[derive(Copy, Clone)]
pub struct ShadowPass;

impl LintPass for ShadowPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SHADOW_SAME, SHADOW_REUSE, SHADOW_FOREIGN)
    }

    fn check_fn(&mut self, cx: &Context, _: FnKind, decl: &FnDecl,
            block: &Block, _: Span, _: NodeId) {
        if in_external_macro(cx, block.span) { return; }
        check_fn(cx, decl, block);
    }
}

fn check_fn(cx: &Context, decl: &FnDecl, block: &Block) {
    let mut bindings = Vec::new();
    for arg in &decl.inputs {
        if let PatIdent(_, ident, _) = arg.pat.node {
            bindings.push(ident.node.name)
        }
    }
    check_block(cx, block, &mut bindings);
}

fn named(pat: &Pat) -> Option<Name> {
    if let PatIdent(_, ident, _) = pat.node {
       Some(ident.node.name)
    } else { None }
}

fn add(bindings: &mut Vec<Name>, pat: &Pat) {
    named(pat).map(|name| bindings.push(name));
}

fn check_block(cx: &Context, block: &Block, bindings: &mut Vec<Name>) {
    let len = bindings.len();
    for stmt in &block.stmts {
        match stmt.node {
            StmtDecl(ref decl, _) => check_decl(cx, decl, bindings),
            StmtExpr(ref e, _) | StmtSemi(ref e, _) =>
                check_expr(cx, e, bindings),
            _ => ()
        }
    }
    if let Some(ref o) = block.expr { check_expr(cx, o, bindings); }
    bindings.truncate(len);
}

fn check_decl(cx: &Context, decl: &Decl, bindings: &mut Vec<Name>) {
    if in_external_macro(cx, decl.span) { return; }
    if let DeclLocal(ref local) = decl.node {
        let Local{ ref pat, ref ty, ref init, id: _, span: _ } = **local;
        if let &Some(ref t) = ty { check_ty(cx, t, bindings); }
        named(pat).map(|name| if bindings.contains(&name) {
            if let &Some(ref o) = init {
                if in_external_macro(cx, o.span) { return; }
                check_expr(cx, o, bindings);
                bindings.push(name);
                lint_shadow(cx, name, decl.span, pat.span, o);
            }
        });
        add(bindings, pat);
        if let &Some(ref o) = init {
            check_expr(cx, o, bindings)
        }
    }
}

fn lint_shadow(cx: &Context, name: Name, span: Span, lspan: Span, init: &Expr) {
    if is_self_shadow(name, init) {
        span_lint(cx, SHADOW_SAME, span, &format!(
            "{} is shadowed by itself in {}",
            snippet(cx, lspan, "_"),
            snippet(cx, init.span, "..")));
    } else {
        if contains_self(name, init) {
            span_lint(cx, SHADOW_REUSE, span, &format!(
                "{} is shadowed by {} which reuses the original value",
                snippet(cx, lspan, "_"),
                snippet(cx, init.span, "..")));
        } else {
            span_lint(cx, SHADOW_FOREIGN, span, &format!(
                "{} is shadowed by {} in this declaration",
                snippet(cx, lspan, "_"),
                snippet(cx, init.span, "..")));
        }
    }
}

fn check_expr(cx: &Context, expr: &Expr, bindings: &mut Vec<Name>) {
    if in_external_macro(cx, expr.span) { return; }
    match expr.node {
        ExprUnary(_, ref e) | ExprParen(ref e) | ExprField(ref e, _) |
        ExprTupField(ref e, _) | ExprAddrOf(_, ref e) | ExprBox(None, ref e)
            => { check_expr(cx, e, bindings) },
        ExprBox(Some(ref place), ref e) => {
            check_expr(cx, place, bindings); check_expr(cx, e, bindings) }
        ExprBlock(ref block) | ExprLoop(ref block, _) =>
            { check_block(cx, block, bindings) },
        ExprVec(ref v) | ExprTup(ref v) =>
            for ref e in v { check_expr(cx, e, bindings) },
        ExprIf(ref cond, ref then, ref otherwise) => {
            check_expr(cx, cond, bindings);
            check_block(cx, then, bindings);
            if let &Some(ref o) = otherwise { check_expr(cx, o, bindings); }
        },
        ExprIfLet(ref pat, ref e, ref block, ref otherwise) => {
            check_expr(cx, e, bindings);
            let len = bindings.len();
            add(bindings, pat);
            check_block(cx, block, bindings);
            if let &Some(ref o) = otherwise { check_expr(cx, o, bindings); }
            bindings.truncate(len);
        },
        ExprWhile(ref cond, ref block, _) => {
            check_expr(cx, cond, bindings);
            check_block(cx, block, bindings);
        },
        ExprWhileLet(ref pat, ref e, ref block, _) |
        ExprForLoop(ref pat, ref e, ref block, _) => {
            check_expr(cx, e, bindings);
            let len = bindings.len();
            add(bindings, pat);
            check_block(cx, block, bindings);
            bindings.truncate(len);
        },
        _ => ()
    }
}

fn check_ty(cx: &Context, ty: &Ty, bindings: &mut Vec<Name>) {
    match ty.node {
        TyParen(ref sty) | TyObjectSum(ref sty, _) |
        TyVec(ref sty) => check_ty(cx, sty, bindings),
        TyFixedLengthVec(ref fty, ref expr) => {
            check_ty(cx, fty, bindings);
            check_expr(cx, expr, bindings);
        },
        TyPtr(MutTy{ ty: ref mty, .. }) |
        TyRptr(_, MutTy{ ty: ref mty, .. }) => check_ty(cx, mty, bindings),
        TyTup(ref tup) => { for ref t in tup { check_ty(cx, t, bindings) } },
        TyTypeof(ref expr) => check_expr(cx, expr, bindings),
        _ => (),
    }
}

fn is_self_shadow(name: Name, expr: &Expr) -> bool {
    match expr.node {
        ExprBox(_, ref inner) |
        ExprParen(ref inner) |
        ExprAddrOf(_, ref inner) => is_self_shadow(name, inner),
        ExprBlock(ref block) => block.stmts.is_empty() && block.expr.as_ref().
            map_or(false, |ref e| is_self_shadow(name, e)),
        ExprUnary(op, ref inner) => (UnUniq == op || UnDeref == op) &&
            is_self_shadow(name, inner),
        ExprPath(_, ref path) => path.segments.len() == 1 &&
            path.segments[0].identifier.name == name,
        _ => false,
    }
}

fn contains_self(name: Name, expr: &Expr) -> bool {
    match expr.node {
        ExprUnary(_, ref e) | ExprParen(ref e) | ExprField(ref e, _) |
        ExprTupField(ref e, _) | ExprAddrOf(_, ref e) | ExprBox(_, ref e)
            => contains_self(name, e),
        ExprBinary(_, ref l, ref r) =>
            contains_self(name, l) || contains_self(name, r),
        ExprBlock(ref block) | ExprLoop(ref block, _) =>
            contains_block_self(name, block),
        ExprCall(ref fun, ref args) => contains_self(name, fun) ||
            args.iter().any(|ref a| contains_self(name, a)),
        ExprMethodCall(_, _, ref args) =>
            args.iter().any(|ref a| contains_self(name, a)),
        ExprVec(ref v) | ExprTup(ref v) =>
            v.iter().any(|ref e| contains_self(name, e)),
        ExprIf(ref cond, ref then, ref otherwise) =>
            contains_self(name, cond) || contains_block_self(name, then) ||
            otherwise.as_ref().map_or(false, |ref e| contains_self(name, e)),
        ExprIfLet(_, ref e, ref block, ref otherwise) =>
            contains_self(name, e) || contains_block_self(name, block) ||
            otherwise.as_ref().map_or(false, |ref o| contains_self(name, o)),
        ExprWhile(ref e, ref block, _) |
        ExprWhileLet(_, ref e, ref block, _) |
        ExprForLoop(_, ref e, ref block, _) =>
            contains_self(name, e) || contains_block_self(name, block),
        ExprPath(_, ref path) => path.segments.len() == 1 &&
            path.segments[0].identifier.name == name,
        _ => false
    }
}

fn contains_block_self(name: Name, block: &Block) -> bool {
    for stmt in &block.stmts {
        match stmt.node {
            StmtDecl(ref decl, _) =>
            if let DeclLocal(ref local) = decl.node {
                if let Some(ref init) = local.init {
                    if contains_self(name, init) { return true; }
                }
            },
            StmtExpr(ref e, _) | StmtSemi(ref e, _) =>
                if contains_self(name, e) { return true },
            _ => ()
        }
    }
    if let Some(ref e) = block.expr { contains_self(name, e) } else { false }
}

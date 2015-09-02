use std::ops::Deref;
use syntax::ast::*;
use syntax::codemap::Span;
use syntax::visit::FnKind;

use rustc::lint::{Context, LintArray, LintPass};
use rustc::middle::def::Def::{DefVariant, DefStruct};

use utils::{in_external_macro, snippet, span_lint, span_note_and_lint};

declare_lint!(pub SHADOW_SAME, Allow,
    "rebinding a name to itself, e.g. `let mut x = &mut x`");
declare_lint!(pub SHADOW_REUSE, Allow,
    "rebinding a name to an expression that re-uses the original value, e.g. \
    `let x = x + 1`");
declare_lint!(pub SHADOW_UNRELATED, Warn,
    "The name is re-bound without even using the original value");

#[derive(Copy, Clone)]
pub struct ShadowPass;

impl LintPass for ShadowPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SHADOW_SAME, SHADOW_REUSE, SHADOW_UNRELATED)
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
        let Local{ ref pat, ref ty, ref init, id: _, span } = **local;
        if let &Some(ref t) = ty { check_ty(cx, t, bindings) }
        if let &Some(ref o) = init {
            check_expr(cx, o, bindings);
            check_pat(cx, pat, &Some(o), span, bindings);
        } else {
            check_pat(cx, pat, &None, span, bindings);
        }
    }
}

fn is_binding(cx: &Context, pat: &Pat) -> bool {
    match cx.tcx.def_map.borrow().get(&pat.id).map(|d| d.full_def()) {
        Some(DefVariant(..)) | Some(DefStruct(..)) => false,
        _ => true
    }
}

fn check_pat(cx: &Context, pat: &Pat, init: &Option<&Expr>, span: Span,
        bindings: &mut Vec<Name>) {
    //TODO: match more stuff / destructuring
    match pat.node {
        PatIdent(_, ref ident, ref inner) => {
            let name = ident.node.name;
            if is_binding(cx, pat) {
                if bindings.contains(&name) {
                    lint_shadow(cx, name, span, pat.span, init);
                } else {
                    bindings.push(name);
                }
            }
            if let Some(ref p) = *inner { check_pat(cx, p, init, span, bindings); }
        },
        //PatEnum(Path, Option<Vec<P<Pat>>>),
        PatStruct(_, ref pfields, _) =>
            if let Some(ref init_struct) = *init {
                if let ExprStruct(_, ref efields, _) = init_struct.node {
                    for field in pfields {
                        let ident = field.node.ident;
                        let efield = efields.iter()
                            .find(|ref f| f.ident.node == ident)
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
                if let ExprBox(_, ref inner_init) = initp.node {
                    check_pat(cx, inner, &Some(&**inner_init), span, bindings);
                } else {
                    check_pat(cx, inner, init, span, bindings);
                }
            } else {
                check_pat(cx, inner, init, span, bindings);
            }
        },
        PatRegion(ref inner, _) =>
            check_pat(cx, inner, init, span, bindings),
        //PatRange(P<Expr>, P<Expr>),
        //PatVec(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>),
        _ => (),
    }
}

fn lint_shadow<T>(cx: &Context, name: Name, span: Span, lspan: Span, init:
        &Option<T>) where T: Deref<Target=Expr> {
    if let Some(ref expr) = *init {
        if is_self_shadow(name, expr) {
            span_lint(cx, SHADOW_SAME, span, &format!(
                "{} is shadowed by itself in {}",
                snippet(cx, lspan, "_"),
                snippet(cx, expr.span, "..")));
        } else {
            if contains_self(name, expr) {
                span_note_and_lint(cx, SHADOW_REUSE, lspan, &format!(
                    "{} is shadowed by {} which reuses the original value",
                    snippet(cx, lspan, "_"),
                    snippet(cx, expr.span, "..")),
                    expr.span, "initialization happens here");
            } else {
                span_note_and_lint(cx, SHADOW_UNRELATED, lspan, &format!(
                    "{} is shadowed by {}",
                    snippet(cx, lspan, "_"),
                    snippet(cx, expr.span, "..")),
                    expr.span, "initialization happens here");
            }
        }
    } else {
        span_lint(cx, SHADOW_UNRELATED, span, &format!(
            "{} shadows a previous declaration", snippet(cx, lspan, "_")));
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
        //ExprCall
        //ExprMethodCall
        ExprVec(ref v) | ExprTup(ref v) =>
            for ref e in v { check_expr(cx, e, bindings) },
        ExprIf(ref cond, ref then, ref otherwise) => {
            check_expr(cx, cond, bindings);
            check_block(cx, then, bindings);
            if let &Some(ref o) = otherwise { check_expr(cx, o, bindings); }
        },
        ExprWhile(ref cond, ref block, _) => {
            check_expr(cx, cond, bindings);
            check_block(cx, block, bindings);
        },
        ExprMatch(ref init, ref arms, _) => {
            check_expr(cx, init, bindings);
            let len = bindings.len();
            for ref arm in arms {
                for ref pat in &arm.pats {
                    check_pat(cx, &pat, &Some(&**init), pat.span, bindings);
                    //TODO: This is ugly, but needed to get the right type
                }
                if let Some(ref guard) = arm.guard {
                    check_expr(cx, guard, bindings);
                }
                check_expr(cx, &arm.body, bindings);
                bindings.truncate(len);
            }
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
        ExprPath(_, ref path) => path_eq_name(name, path),
        _ => false,
    }
}

fn path_eq_name(name: Name, path: &Path) -> bool {
    !path.global && path.segments.len() == 1 &&
        path.segments[0].identifier.name == name
}

fn contains_self(name: Name, expr: &Expr) -> bool {
    match expr.node {
        // the "self" name itself (maybe)
        ExprPath(_, ref path) => path_eq_name(name, path),
        // no subexprs
        ExprLit(_) => false,
        // one subexpr
        ExprUnary(_, ref e) | ExprParen(ref e) | ExprField(ref e, _) |
        ExprTupField(ref e, _) | ExprAddrOf(_, ref e) | ExprBox(_, ref e) |
        ExprCast(ref e, _) =>
            contains_self(name, e),
        // two subexprs
        ExprBinary(_, ref l, ref r) | ExprIndex(ref l, ref r) |
        ExprAssign(ref l, ref r) | ExprAssignOp(_, ref l, ref r) |
        ExprRepeat(ref l, ref r) =>
            contains_self(name, l) || contains_self(name, r),
        // one optional subexpr
        ExprRet(ref oe) =>
            oe.as_ref().map_or(false, |ref e| contains_self(name, e)),
        // two optional subexprs
        ExprRange(ref ol, ref or) =>
            ol.as_ref().map_or(false, |ref e| contains_self(name, e)) ||
            or.as_ref().map_or(false, |ref e| contains_self(name, e)),
        // one subblock
        ExprBlock(ref block) | ExprLoop(ref block, _) |
        ExprClosure(_, _, ref block) =>
            contains_block_self(name, block),
        // one vec
        ExprMethodCall(_, _, ref v) | ExprVec(ref v) | ExprTup(ref v) =>
            v.iter().any(|ref a| contains_self(name, a)),
        // one expr, one vec
        ExprCall(ref fun, ref args) =>
            contains_self(name, fun) ||
            args.iter().any(|ref a| contains_self(name, a)),
        // special ones
        ExprIf(ref cond, ref then, ref otherwise) =>
            contains_self(name, cond) || contains_block_self(name, then) ||
            otherwise.as_ref().map_or(false, |ref e| contains_self(name, e)),
        ExprWhile(ref e, ref block, _)  =>
            contains_self(name, e) || contains_block_self(name, block),
        ExprMatch(ref e, ref arms, _) =>
            contains_self(name, e) ||
            arms.iter().any(
                |ref arm|
                arm.pats.iter().any(|ref pat| contains_pat_self(name, pat)) ||
                arm.guard.as_ref().map_or(false, |ref g| contains_self(name, g)) ||
                contains_self(name, &arm.body)),
        ExprStruct(_, ref fields, ref other) =>
            fields.iter().any(|ref f| contains_self(name, &f.expr)) ||
            other.as_ref().map_or(false, |ref e| contains_self(name, e)),
        _ => false,
    }
}

fn contains_block_self(name: Name, block: &Block) -> bool {
    for stmt in &block.stmts {
        match stmt.node {
            StmtDecl(ref decl, _) =>
            if let DeclLocal(ref local) = decl.node {
                //TODO: We don't currently handle the case where the name
                //is shadowed wiithin the block; this means code including this
                //degenerate pattern will get the wrong warning.
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

fn contains_pat_self(name: Name, pat: &Pat) -> bool {
    match pat.node {
        PatIdent(_, ref ident, ref inner) => name == ident.node.name ||
            inner.as_ref().map_or(false, |ref p| contains_pat_self(name, p)),
        PatEnum(_, ref opats) => opats.as_ref().map_or(false,
            |pats| pats.iter().any(|p| contains_pat_self(name, p))),
        PatQPath(_, ref path) => path_eq_name(name, path),
        PatStruct(_, ref fieldpats, _) => fieldpats.iter().any(
            |ref fp| contains_pat_self(name, &fp.node.pat)),
        PatTup(ref ps) => ps.iter().any(|ref p| contains_pat_self(name, p)),
        PatBox(ref p) |
        PatRegion(ref p, _) => contains_pat_self(name, p),
        PatRange(ref from, ref until) =>
            contains_self(name, from) || contains_self(name, until),
        PatVec(ref pre, ref opt, ref post) =>
            pre.iter().any(|ref p| contains_pat_self(name, p)) ||
                opt.as_ref().map_or(false, |ref p| contains_pat_self(name, p)) ||
                post.iter().any(|ref p| contains_pat_self(name, p)),
        _ => false,
    }
}

use rustc::lint::*;
use rustc::hir::*;
use syntax::ast;
use utils::{is_adjusted, match_path, match_trait_method, match_type, paths, snippet,
            span_help_and_lint, walk_ptrs_ty, walk_ptrs_ty_depth};

/// **What it does:** This lint checks for mapping clone() over an iterator.
///
/// **Why is this bad?** It makes the code less readable.
///
/// **Known problems:** None
///
/// **Example:** `x.map(|e| e.clone());`
declare_lint! {
    pub MAP_CLONE, Warn,
    "using `.map(|x| x.clone())` to clone an iterator or option's contents (recommends \
     `.cloned()` instead)"
}

#[derive(Copy, Clone)]
pub struct MapClonePass;

impl LateLintPass for MapClonePass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        // call to .map()
        if let ExprMethodCall(name, _, ref args) = expr.node {
            if name.node.as_str() == "map" && args.len() == 2 {
                match args[1].node {
                    ExprClosure(_, ref decl, ref blk, _) => {
                        if_let_chain! {[
                            // just one expression in the closure
                            blk.stmts.is_empty(),
                            let Some(ref closure_expr) = blk.expr,
                            // nothing special in the argument, besides reference bindings
                            // (e.g. .map(|&x| x) )
                            let Some(arg_ident) = get_arg_name(&*decl.inputs[0].pat),
                            // the method is being called on a known type (option or iterator)
                            let Some(type_name) = get_type_name(cx, expr, &args[0])
                        ], {
                            // look for derefs, for .map(|x| *x)
                            if only_derefs(cx, &*closure_expr, arg_ident) &&
                                // .cloned() only removes one level of indirection, don't lint on more
                                walk_ptrs_ty_depth(cx.tcx.pat_ty(&*decl.inputs[0].pat)).1 == 1
                            {
                                span_help_and_lint(cx, MAP_CLONE, expr.span, &format!(
                                    "you seem to be using .map() to clone the contents of an {}, consider \
                                    using `.cloned()`", type_name),
                                    &format!("try\n{}.cloned()", snippet(cx, args[0].span, "..")));
                            }
                            // explicit clone() calls ( .map(|x| x.clone()) )
                            else if let ExprMethodCall(clone_call, _, ref clone_args) = closure_expr.node {
                                if clone_call.node.as_str() == "clone" &&
                                    clone_args.len() == 1 &&
                                    match_trait_method(cx, closure_expr, &paths::CLONE_TRAIT) &&
                                    expr_eq_name(&clone_args[0], arg_ident)
                                {
                                    span_help_and_lint(cx, MAP_CLONE, expr.span, &format!(
                                        "you seem to be using .map() to clone the contents of an {}, consider \
                                        using `.cloned()`", type_name),
                                        &format!("try\n{}.cloned()", snippet(cx, args[0].span, "..")));
                                }
                            }
                        }}
                    }
                    ExprPath(_, ref path) => {
                        if match_path(path, &paths::CLONE) {
                            let type_name = get_type_name(cx, expr, &args[0]).unwrap_or("_");
                            span_help_and_lint(cx,
                                               MAP_CLONE,
                                               expr.span,
                                               &format!("you seem to be using .map() to clone the contents of an \
                                                         {}, consider using `.cloned()`",
                                                        type_name),
                                               &format!("try\n{}.cloned()", snippet(cx, args[0].span, "..")));
                        }
                    }
                    _ => (),
                }
            }
        }
    }
}

fn expr_eq_name(expr: &Expr, id: ast::Name) -> bool {
    match expr.node {
        ExprPath(None, ref path) => {
            let arg_segment = [PathSegment {
                                   name: id,
                                   parameters: PathParameters::none(),
                               }];
            !path.global && path.segments[..] == arg_segment
        }
        _ => false,
    }
}

fn get_type_name(cx: &LateContext, expr: &Expr, arg: &Expr) -> Option<&'static str> {
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        Some("iterator")
    } else if match_type(cx, walk_ptrs_ty(cx.tcx.expr_ty(arg)), &paths::OPTION) {
        Some("Option")
    } else {
        None
    }
}

fn get_arg_name(pat: &Pat) -> Option<ast::Name> {
    match pat.node {
        PatKind::Binding(_, name, None) => Some(name.node),
        PatKind::Ref(ref subpat, _) => get_arg_name(subpat),
        _ => None,
    }
}

fn only_derefs(cx: &LateContext, expr: &Expr, id: ast::Name) -> bool {
    match expr.node {
        ExprUnary(UnDeref, ref subexpr) if !is_adjusted(cx, subexpr) => only_derefs(cx, subexpr, id),
        _ => expr_eq_name(expr, id),
    }
}

impl LintPass for MapClonePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MAP_CLONE)
    }
}

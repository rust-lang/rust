use rustc::lint::*;
use rustc::hir::*;
use rustc::ty;
use syntax::ast;
use utils::{is_adjusted, iter_input_pats, match_qpath, match_trait_method, match_type, paths, remove_blocks, snippet,
            span_help_and_lint, walk_ptrs_ty, walk_ptrs_ty_depth};

/// **What it does:** Checks for mapping `clone()` over an iterator.
///
/// **Why is this bad?** It makes the code less readable than using the
/// `.cloned()` adapter.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// x.map(|e| e.clone());
/// ```
declare_lint! {
    pub MAP_CLONE,
    Warn,
    "using `.map(|x| x.clone())` to clone an iterator or option's contents"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        // call to .map()
        if let ExprMethodCall(ref method, _, ref args) = expr.node {
            if method.name == "map" && args.len() == 2 {
                match args[1].node {
                    ExprClosure(_, ref decl, closure_eid, _, _) => {
                        let body = cx.tcx.hir.body(closure_eid);
                        let closure_expr = remove_blocks(&body.value);
                        let ty = cx.tables.pat_ty(&body.arguments[0].pat);
                        if_let_chain! {[
                            // nothing special in the argument, besides reference bindings
                            // (e.g. .map(|&x| x) )
                            let Some(first_arg) = iter_input_pats(decl, body).next(),
                            let Some(arg_ident) = get_arg_name(&first_arg.pat),
                            // the method is being called on a known type (option or iterator)
                            let Some(type_name) = get_type_name(cx, expr, &args[0])
                        ], {
                            // look for derefs, for .map(|x| *x)
                            if only_derefs(cx, &*closure_expr, arg_ident) &&
                                // .cloned() only removes one level of indirection, don't lint on more
                                walk_ptrs_ty_depth(cx.tables.pat_ty(&first_arg.pat)).1 == 1
                            {
                                // the argument is not an &mut T
                                if let ty::TyRef(_, tam) = ty.sty {
                                    if tam.mutbl == MutImmutable {
                                        span_help_and_lint(cx, MAP_CLONE, expr.span, &format!(
                                            "you seem to be using .map() to clone the contents of an {}, consider \
                                            using `.cloned()`", type_name),
                                            &format!("try\n{}.cloned()", snippet(cx, args[0].span, "..")));
                                    }
                                }
                            }
                            // explicit clone() calls ( .map(|x| x.clone()) )
                            else if let ExprMethodCall(ref clone_call, _, ref clone_args) = closure_expr.node {
                                if clone_call.name == "clone" &&
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
                    },
                    ExprPath(ref path) => if match_qpath(path, &paths::CLONE) {
                        let type_name = get_type_name(cx, expr, &args[0]).unwrap_or("_");
                        span_help_and_lint(
                            cx,
                            MAP_CLONE,
                            expr.span,
                            &format!(
                                "you seem to be using .map() to clone the contents of an \
                                 {}, consider using `.cloned()`",
                                type_name
                            ),
                            &format!("try\n{}.cloned()", snippet(cx, args[0].span, "..")),
                        );
                    },
                    _ => (),
                }
            }
        }
    }
}

fn expr_eq_name(expr: &Expr, id: ast::Name) -> bool {
    match expr.node {
        ExprPath(QPath::Resolved(None, ref path)) => {
            let arg_segment = [
                PathSegment {
                    name: id,
                    parameters: PathParameters::none(),
                },
            ];
            !path.is_global() && path.segments[..] == arg_segment
        },
        _ => false,
    }
}

fn get_type_name(cx: &LateContext, expr: &Expr, arg: &Expr) -> Option<&'static str> {
    if match_trait_method(cx, expr, &paths::ITERATOR) {
        Some("iterator")
    } else if match_type(cx, walk_ptrs_ty(cx.tables.expr_ty(arg)), &paths::OPTION) {
        Some("Option")
    } else {
        None
    }
}

fn get_arg_name(pat: &Pat) -> Option<ast::Name> {
    match pat.node {
        PatKind::Binding(_, _, name, None) => Some(name.node),
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

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MAP_CLONE)
    }
}

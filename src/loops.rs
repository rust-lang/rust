use rustc::lint::*;
use syntax::ast::*;
use syntax::visit::{Visitor, walk_expr};
use std::collections::HashSet;

use utils::{snippet, span_lint, get_parent_expr};

declare_lint!{ pub NEEDLESS_RANGE_LOOP, Warn,
               "for-looping over a range of indices where an iterator over items would do" }

declare_lint!{ pub EXPLICIT_ITER_LOOP, Warn,
               "for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do" }

#[derive(Copy, Clone)]
pub struct LoopsPass;

impl LintPass for LoopsPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_RANGE_LOOP, EXPLICIT_ITER_LOOP)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let Some((pat, arg, body)) = recover_for_loop(expr) {
            // check for looping over a range and then indexing a sequence with it
            // -> the iteratee must be a range literal
            if let ExprRange(_, _) = arg.node {
                // the var must be a single name
                if let PatIdent(_, ref ident, _) = pat.node {
                    let mut visitor = VarVisitor { cx: cx, var: ident.node.name,
                                                   indexed: HashSet::new(), nonindex: false };
                    walk_expr(&mut visitor, body);
                    // linting condition: we only indexed one variable
                    if visitor.indexed.len() == 1 {
                        let indexed = visitor.indexed.into_iter().next().expect("Len was nonzero, but no contents found");
                        if visitor.nonindex {
                            span_lint(cx, NEEDLESS_RANGE_LOOP, expr.span, &format!(
                                "the loop variable `{}` is used to index `{}`. Consider using \
                                 `for ({}, item) in {}.iter().enumerate()` or similar iterators",
                                ident.node.name, indexed, ident.node.name, indexed));
                        } else {
                            span_lint(cx, NEEDLESS_RANGE_LOOP, expr.span, &format!(
                                "the loop variable `{}` is only used to index `{}`. \
                                 Consider using `for item in &{}` or similar iterators",
                                ident.node.name, indexed, indexed));
                        }
                    }
                }
            }

            // check for looping over x.iter() or x.iter_mut(), could use &x or &mut x
            if let ExprMethodCall(ref method, _, ref args) = arg.node {
                // just the receiver, no arguments to iter() or iter_mut()
                if args.len() == 1 {
                    let method_name = method.node.name;
                    if method_name == "iter" {
                        let object = snippet(cx, args[0].span, "_");
                        span_lint(cx, EXPLICIT_ITER_LOOP, expr.span, &format!(
                            "it is more idiomatic to loop over `&{}` instead of `{}.iter()`",
                            object, object));
                    } else if method_name == "iter_mut" {
                        let object = snippet(cx, args[0].span, "_");
                        span_lint(cx, EXPLICIT_ITER_LOOP, expr.span, &format!(
                            "it is more idiomatic to loop over `&mut {}` instead of `{}.iter_mut()`",
                            object, object));
                    }
                }
            }
        }
    }
}

/// Recover the essential nodes of a desugared for loop:
/// `for pat in arg { body }` becomes `(pat, arg, body)`.
fn recover_for_loop(expr: &Expr) -> Option<(&Pat, &Expr, &Expr)> {
    if_let_chain! {
        [
            let ExprMatch(ref iterexpr, ref arms, _) = expr.node,
            let ExprCall(_, ref iterargs) = iterexpr.node,
            iterargs.len() == 1 && arms.len() == 1 && arms[0].guard.is_none(),
            let ExprLoop(ref block, _) = arms[0].body.node,
            block.stmts.is_empty(),
            let Some(ref loopexpr) = block.expr,
            let ExprMatch(_, ref innerarms, MatchSource::ForLoopDesugar) = loopexpr.node,
            innerarms.len() == 2 && innerarms[0].pats.len() == 1,
            let PatEnum(_, Some(ref somepats)) = innerarms[0].pats[0].node,
            somepats.len() == 1
        ], {
            return Some((&*somepats[0],
                         &*iterargs[0],
                         &*innerarms[0].body));
        }
    }
    None
}

struct VarVisitor<'v, 't: 'v> {
    cx: &'v Context<'v, 't>, // context reference
    var: Name,               // var name to look for as index
    indexed: HashSet<Name>,  // indexed variables
    nonindex: bool,          // has the var been used otherwise?
}

impl<'v, 't> Visitor<'v> for VarVisitor<'v, 't> {
    fn visit_expr(&mut self, expr: &'v Expr) {
        if let ExprPath(None, ref path) = expr.node {
            if path.segments.len() == 1 && path.segments[0].identifier.name == self.var {
                // we are referencing our variable! now check if it's as an index
                if_let_chain! {
                    [
                        let Some(parexpr) = get_parent_expr(self.cx, expr),
                        let ExprIndex(ref seqexpr, _) = parexpr.node,
                        let ExprPath(None, ref seqvar) = seqexpr.node,
                        seqvar.segments.len() == 1
                    ], {
                        self.indexed.insert(seqvar.segments[0].identifier.name);
                        return;  // no need to walk further
                    }
                }
                // we are not indexing anything, record that
                self.nonindex = true;
                return;
            }
        }
        walk_expr(self, expr);
    }
}

use rustc::lint::*;
use syntax::ast::*;
use syntax::visit::{Visitor, walk_expr};
use rustc::middle::ty;
use std::collections::HashSet;

use utils::{snippet, span_lint, get_parent_expr, match_trait_method, match_type, walk_ptrs_ty,
            in_external_macro, expr_block, span_help_and_lint};
use utils::{VEC_PATH, LL_PATH};

declare_lint!{ pub NEEDLESS_RANGE_LOOP, Warn,
               "for-looping over a range of indices where an iterator over items would do" }

declare_lint!{ pub EXPLICIT_ITER_LOOP, Warn,
               "for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do" }

declare_lint!{ pub ITER_NEXT_LOOP, Warn,
               "for-looping over `_.next()` which is probably not intended" }

declare_lint!{ pub WHILE_LET_LOOP, Warn,
               "`loop { if let { ... } else break }` can be written as a `while let` loop" }

#[derive(Copy, Clone)]
pub struct LoopsPass;

impl LintPass for LoopsPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_RANGE_LOOP, EXPLICIT_ITER_LOOP, ITER_NEXT_LOOP,
                    WHILE_LET_LOOP)
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
                        let indexed = visitor.indexed.into_iter().next().expect(
                            "Len was nonzero, but no contents found");
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

            if let ExprMethodCall(ref method, _, ref args) = arg.node {
                // just the receiver, no arguments
                if args.len() == 1 {
                    let method_name = method.node.name;
                    // check for looping over x.iter() or x.iter_mut(), could use &x or &mut x
                    if method_name == "iter" || method_name == "iter_mut" {
                        if is_ref_iterable_type(cx, &args[0]) {
                            let object = snippet(cx, args[0].span, "_");
                            span_lint(cx, EXPLICIT_ITER_LOOP, expr.span, &format!(
                                "it is more idiomatic to loop over `&{}{}` instead of `{}.{}()`",
                                if method_name == "iter_mut" { "mut " } else { "" },
                                object, object, method_name));
                        }
                    }
                    // check for looping over Iterator::next() which is not what you want
                    else if method_name == "next" {
                        if match_trait_method(cx, arg, &["core", "iter", "Iterator"]) {
                            span_lint(cx, ITER_NEXT_LOOP, expr.span,
                                      "you are iterating over `Iterator::next()` which is an Option; \
                                       this will compile but is probably not what you want");
                        }
                    }
                }
            }
        }
        // check for `loop { if let {} else break }` that could be `while let`
        // (also matches explicit "match" instead of "if let")
        if let ExprLoop(ref block, _) = expr.node {
            // extract a single expression
            if let Some(inner) = extract_single_expr(block) {
                if let ExprMatch(ref matchexpr, ref arms, ref source) = inner.node {
                    // ensure "if let" compatible match structure
                    match *source {
                        MatchSource::Normal | MatchSource::IfLetDesugar{..} => if
                            arms.len() == 2 &&
                            arms[0].pats.len() == 1 && arms[0].guard.is_none() &&
                            arms[1].pats.len() == 1 && arms[1].guard.is_none() &&
                            // finally, check for "break" in the second clause
                            is_break_expr(&arms[1].body)
                        {
                            if in_external_macro(cx, expr.span) { return; }
                            span_help_and_lint(cx, WHILE_LET_LOOP, expr.span,
                                               "this loop could be written as a `while let` loop",
                                               &format!("try\nwhile let {} = {} {}",
                                                        snippet(cx, arms[0].pats[0].span, ".."),
                                                        snippet(cx, matchexpr.span, ".."),
                                                        expr_block(cx, &arms[0].body, "..")));
                        },
                        _ => ()
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
            return Some((&somepats[0],
                         &iterargs[0],
                         &innerarms[0].body));
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

/// Return true if the type of expr is one that provides IntoIterator impls
/// for &T and &mut T, such as Vec.
fn is_ref_iterable_type(cx: &Context, e: &Expr) -> bool {
    let ty = walk_ptrs_ty(cx.tcx.expr_ty(e));
    println!("mt {:?} {:?}", e, ty);
    is_array(ty) ||
        match_type(cx, ty, &VEC_PATH) ||
        match_type(cx, ty, &LL_PATH) ||
        match_type(cx, ty, &["std", "collections", "hash", "map", "HashMap"]) ||
        match_type(cx, ty, &["std", "collections", "hash", "set", "HashSet"]) ||
        match_type(cx, ty, &["collections", "vec_deque", "VecDeque"]) ||
        match_type(cx, ty, &["collections", "binary_heap", "BinaryHeap"]) ||
        match_type(cx, ty, &["collections", "btree", "map", "BTreeMap"]) ||
        match_type(cx, ty, &["collections", "btree", "set", "BTreeSet"])
}

fn is_array(ty: ty::Ty) -> bool {
    match ty.sty {
        ty::TyArray(..) => true,
        _ => false
    }
}

/// If block consists of a single expression (with or without semicolon), return it.
fn extract_single_expr(block: &Block) -> Option<&Expr> {
    match (&block.stmts.len(), &block.expr) {
        (&1, &None) => match block.stmts[0].node {
            StmtExpr(ref expr, _) |
            StmtSemi(ref expr, _) => Some(expr),
            _ => None,
        },
        (&0, &Some(ref expr)) => Some(expr),
        _ => None
    }
}

/// Return true if expr contains a single break expr (maybe within a block).
fn is_break_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprBreak(None) => true,
        ExprBlock(ref b) => match extract_single_expr(b) {
            Some(ref subexpr) => is_break_expr(subexpr),
            None => false,
        },
        _ => false,
    }
}

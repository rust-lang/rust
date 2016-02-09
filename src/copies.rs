use rustc::lint::*;
use rustc_front::hir::*;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use utils::{SpanlessEq, SpanlessHash};
use utils::{get_parent_expr, in_macro, span_note_and_lint};

/// **What it does:** This lint checks for consecutive `ifs` with the same condition. This lint is
/// `Warn` by default.
///
/// **Why is this bad?** This is probably a copy & paste error.
///
/// **Known problems:** Hopefully none.
///
/// **Example:** `if a == b { .. } else if a == b { .. }`
declare_lint! {
    pub IFS_SAME_COND,
    Warn,
    "consecutive `ifs` with the same condition"
}

/// **What it does:** This lint checks for `if/else` with the same body as the *then* part and the
/// *else* part. This lint is `Warn` by default.
///
/// **Why is this bad?** This is probably a copy & paste error.
///
/// **Known problems:** Hopefully none.
///
/// **Example:** `if .. { 42 } else { 42 }`
declare_lint! {
    pub IF_SAME_THEN_ELSE,
    Warn,
    "if with the same *then* and *else* blocks"
}

#[derive(Copy, Clone, Debug)]
pub struct CopyAndPaste;

impl LintPass for CopyAndPaste {
    fn get_lints(&self) -> LintArray {
        lint_array![
            IFS_SAME_COND,
            IF_SAME_THEN_ELSE
        ]
    }
}

impl LateLintPass for CopyAndPaste {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if !in_macro(cx, expr.span) {
            // skip ifs directly in else, it will be checked in the parent if
            if let Some(&Expr{node: ExprIf(_, _, Some(ref else_expr)), ..}) = get_parent_expr(cx, expr) {
                if else_expr.id == expr.id {
                    return;
                }
            }

            let (conds, blocks) = if_sequence(expr);
            lint_same_then_else(cx, &blocks);
            lint_same_cond(cx, &conds);
        }
    }
}

/// Implementation of `IF_SAME_THEN_ELSE`.
fn lint_same_then_else(cx: &LateContext, blocks: &[&Block]) {
    let hash = |block| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_block(block);
        h.finish()
    };
    let eq = |lhs, rhs| -> bool {
        SpanlessEq::new(cx).eq_block(lhs, rhs)
    };

    if let Some((i, j)) = search_same(blocks, hash, eq) {
        span_note_and_lint(cx, IF_SAME_THEN_ELSE, j.span, "this if has identical blocks", i.span, "same as this");
    }
}

/// Implementation of `IFS_SAME_COND`.
fn lint_same_cond(cx: &LateContext, conds: &[&Expr]) {
    let hash = |expr| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_expr(expr);
        h.finish()
    };
    let eq = |lhs, rhs| -> bool {
        SpanlessEq::new(cx).ignore_fn().eq_expr(lhs, rhs)
    };

    if let Some((i, j)) = search_same(conds, hash, eq) {
        span_note_and_lint(cx, IFS_SAME_COND, j.span, "this if has the same condition as a previous if", i.span, "same as this");
    }
}

/// Return the list of condition expressions and the list of blocks in a sequence of `if/else`.
/// Eg. would return `([a, b], [c, d, e])` for the expression
/// `if a { c } else if b { d } else { e }`.
fn if_sequence(mut expr: &Expr) -> (Vec<&Expr>, Vec<&Block>) {
    let mut conds = vec![];
    let mut blocks = vec![];

    while let ExprIf(ref cond, ref then_block, ref else_expr) = expr.node {
        conds.push(&**cond);
        blocks.push(&**then_block);

        if let Some(ref else_expr) = *else_expr {
            expr = else_expr;
        }
        else {
            break;
        }
    }

    // final `else {..}`
    if !blocks.is_empty() {
        if let ExprBlock(ref block) = expr.node {
            blocks.push(&**block);
        }
    }

    (conds, blocks)
}

fn search_same<'a, T, Hash, Eq>(exprs: &[&'a T],
                                hash: Hash,
                                eq: Eq) -> Option<(&'a T, &'a T)>
where Hash: Fn(&'a T) -> u64,
      Eq: Fn(&'a T, &'a T) -> bool {
    // common cases
    if exprs.len() < 2 {
        return None;
    }
    else if exprs.len() == 2 {
        return if eq(&exprs[0], &exprs[1]) {
            Some((&exprs[0], &exprs[1]))
        }
        else {
            None
        }
    }

    let mut map : HashMap<_, Vec<&'a _>> = HashMap::with_capacity(exprs.len());

    for &expr in exprs {
        match map.entry(hash(expr)) {
            Entry::Occupied(o) => {
                for o in o.get() {
                    if eq(o, expr) {
                        return Some((o, expr))
                    }
                }
            }
            Entry::Vacant(v) => { v.insert(vec![expr]); }
        }
    }

    None
}

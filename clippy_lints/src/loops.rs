use itertools::Itertools;
use reexport::*;
use rustc::hir::*;
use rustc::hir::def::Def;
use rustc::hir::intravisit::{walk_block, walk_decl, walk_expr, walk_pat, walk_stmt, NestedVisitorMap, Visitor};
use rustc::hir::map::Node::{NodeBlock, NodeExpr, NodeStmt};
use rustc::lint::*;
use rustc::middle::const_val::ConstVal;
use rustc::middle::region;
use rustc::middle::region::CodeExtent;
use rustc::middle::expr_use_visitor::*;
use rustc::middle::mem_categorization::cmt;
use rustc::ty::{self, Ty};
use rustc::ty::subst::{Subst, Substs};
use rustc_const_eval::ConstContext;
use std::collections::{HashMap, HashSet};
use syntax::ast;
use syntax::codemap::Span;
use utils::sugg;
use utils::const_to_u64;

use utils::{get_enclosing_block, get_parent_expr, higher, in_external_macro, is_integer_literal, is_refutable,
            last_path_segment, match_trait_method, match_type, match_var, multispan_sugg, snippet, snippet_opt,
            span_help_and_lint, span_lint, span_lint_and_sugg, span_lint_and_then};
use utils::paths;

/// **What it does:** Checks for for-loops that manually copy items between
/// slices that could be optimized by having a memcpy.
///
/// **Why is this bad?** It is not as fast as a memcpy.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for i in 0..src.len() {
///     dst[i + 64] = src[i];
/// }
/// ```
declare_lint! {
    pub MANUAL_MEMCPY,
    Warn,
    "manually copying items between slices"
}

/// **What it does:** Checks for looping over the range of `0..len` of some
/// collection just to get the values by index.
///
/// **Why is this bad?** Just iterating the collection itself makes the intent
/// more clear and is probably faster.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for i in 0..vec.len() {
///     println!("{}", vec[i]);
/// }
/// ```
declare_lint! {
    pub NEEDLESS_RANGE_LOOP,
    Warn,
    "for-looping over a range of indices where an iterator over items would do"
}

/// **What it does:** Checks for loops on `x.iter()` where `&x` will do, and
/// suggests the latter.
///
/// **Why is this bad?** Readability.
///
/// **Known problems:** False negatives. We currently only warn on some known
/// types.
///
/// **Example:**
/// ```rust
/// // with `y` a `Vec` or slice:
/// for x in y.iter() { .. }
/// ```
declare_lint! {
    pub EXPLICIT_ITER_LOOP,
    Warn,
    "for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do"
}

/// **What it does:** Checks for loops on `y.into_iter()` where `y` will do, and
/// suggests the latter.
///
/// **Why is this bad?** Readability.
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// // with `y` a `Vec` or slice:
/// for x in y.into_iter() { .. }
/// ```
declare_lint! {
    pub EXPLICIT_INTO_ITER_LOOP,
    Warn,
    "for-looping over `_.into_iter()` when `_` would do"
}

/// **What it does:** Checks for loops on `x.next()`.
///
/// **Why is this bad?** `next()` returns either `Some(value)` if there was a
/// value, or `None` otherwise. The insidious thing is that `Option<_>`
/// implements `IntoIterator`, so that possibly one value will be iterated,
/// leading to some hard to find bugs. No one will want to write such code
/// [except to win an Underhanded Rust
/// Contest](https://www.reddit.
/// com/r/rust/comments/3hb0wm/underhanded_rust_contest/cu5yuhr).
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for x in y.next() { .. }
/// ```
declare_lint! {
    pub ITER_NEXT_LOOP,
    Warn,
    "for-looping over `_.next()` which is probably not intended"
}

/// **What it does:** Checks for `for` loops over `Option` values.
///
/// **Why is this bad?** Readability. This is more clearly expressed as an `if
/// let`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for x in option { .. }
/// ```
///
/// This should be
/// ```rust
/// if let Some(x) = option { .. }
/// ```
declare_lint! {
    pub FOR_LOOP_OVER_OPTION,
    Warn,
    "for-looping over an `Option`, which is more clearly expressed as an `if let`"
}

/// **What it does:** Checks for `for` loops over `Result` values.
///
/// **Why is this bad?** Readability. This is more clearly expressed as an `if
/// let`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for x in result { .. }
/// ```
///
/// This should be
/// ```rust
/// if let Ok(x) = result { .. }
/// ```
declare_lint! {
    pub FOR_LOOP_OVER_RESULT,
    Warn,
    "for-looping over a `Result`, which is more clearly expressed as an `if let`"
}

/// **What it does:** Detects `loop + match` combinations that are easier
/// written as a `while let` loop.
///
/// **Why is this bad?** The `while let` loop is usually shorter and more
/// readable.
///
/// **Known problems:** Sometimes the wrong binding is displayed (#383).
///
/// **Example:**
/// ```rust
/// loop {
///     let x = match y {
///         Some(x) => x,
///         None => break,
///     }
///     // .. do something with x
/// }
/// // is easier written as
/// while let Some(x) = y {
///     // .. do something with x
/// }
/// ```
declare_lint! {
    pub WHILE_LET_LOOP,
    Warn,
    "`loop { if let { ... } else break }`, which can be written as a `while let` loop"
}

/// **What it does:** Checks for using `collect()` on an iterator without using
/// the result.
///
/// **Why is this bad?** It is more idiomatic to use a `for` loop over the
/// iterator instead.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// vec.iter().map(|x| /* some operation returning () */).collect::<Vec<_>>();
/// ```
declare_lint! {
    pub UNUSED_COLLECT,
    Warn,
    "`collect()`ing an iterator without using the result; this is usually better \
     written as a for loop"
}

/// **What it does:** Checks for loops over ranges `x..y` where both `x` and `y`
/// are constant and `x` is greater or equal to `y`, unless the range is
/// reversed or has a negative `.step_by(_)`.
///
/// **Why is it bad?** Such loops will either be skipped or loop until
/// wrap-around (in debug code, this may `panic!()`). Both options are probably
/// not intended.
///
/// **Known problems:** The lint cannot catch loops over dynamically defined
/// ranges. Doing this would require simulating all possible inputs and code
/// paths through the program, which would be complex and error-prone.
///
/// **Example:**
/// ```rust
/// for x in 5..10-5 { .. } // oops, stray `-`
/// ```
declare_lint! {
    pub REVERSE_RANGE_LOOP,
    Warn,
    "iteration over an empty range, such as `10..0` or `5..5`"
}

/// **What it does:** Checks `for` loops over slices with an explicit counter
/// and suggests the use of `.enumerate()`.
///
/// **Why is it bad?** Not only is the version using `.enumerate()` more
/// readable, the compiler is able to remove bounds checks which can lead to
/// faster code in some instances.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for i in 0..v.len() { foo(v[i]);
/// for i in 0..v.len() { bar(i, v[i]); }
/// ```
declare_lint! {
    pub EXPLICIT_COUNTER_LOOP,
    Warn,
    "for-looping with an explicit counter when `_.enumerate()` would do"
}

/// **What it does:** Checks for empty `loop` expressions.
///
/// **Why is this bad?** Those busy loops burn CPU cycles without doing
/// anything. Think of the environment and either block on something or at least
/// make the thread sleep for some microseconds.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// loop {}
/// ```
declare_lint! {
    pub EMPTY_LOOP,
    Warn,
    "empty `loop {}`, which should block or sleep"
}

/// **What it does:** Checks for `while let` expressions on iterators.
///
/// **Why is this bad?** Readability. A simple `for` loop is shorter and conveys
/// the intent better.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// while let Some(val) = iter() { .. }
/// ```
declare_lint! {
    pub WHILE_LET_ON_ITERATOR,
    Warn,
    "using a while-let loop instead of a for loop on an iterator"
}

/// **What it does:** Checks for iterating a map (`HashMap` or `BTreeMap`) and
/// ignoring either the keys or values.
///
/// **Why is this bad?** Readability. There are `keys` and `values` methods that
/// can be used to express that don't need the values or keys.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for (k, _) in &map { .. }
/// ```
///
/// could be replaced by
///
/// ```rust
/// for k in map.keys() { .. }
/// ```
declare_lint! {
    pub FOR_KV_MAP,
    Warn,
    "looping on a map using `iter` when `keys` or `values` would do"
}

/// **What it does:** Checks for loops that will always `break`, `return` or
/// `continue` an outer loop.
///
/// **Why is this bad?** This loop never loops, all it does is obfuscating the
/// code.
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// loop { ..; break; }
/// ```
declare_lint! {
    pub NEVER_LOOP,
    Warn,
    "any loop that will always `break` or `return`"
}

/// TODO: add documentation

declare_lint! {
    pub MUT_RANGE_BOUND,
    Warn,
    "for loop over a range where one of the bounds is a mutable variable"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            MANUAL_MEMCPY,
            NEEDLESS_RANGE_LOOP,
            EXPLICIT_ITER_LOOP,
            EXPLICIT_INTO_ITER_LOOP,
            ITER_NEXT_LOOP,
            FOR_LOOP_OVER_RESULT,
            FOR_LOOP_OVER_OPTION,
            WHILE_LET_LOOP,
            UNUSED_COLLECT,
            REVERSE_RANGE_LOOP,
            EXPLICIT_COUNTER_LOOP,
            EMPTY_LOOP,
            WHILE_LET_ON_ITERATOR,
            FOR_KV_MAP,
            NEVER_LOOP, 
            MUT_RANGE_BOUND
        )
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let Some((pat, arg, body)) = higher::for_loop(expr) {
            check_for_loop(cx, pat, arg, body, expr);
        }

        // check for never_loop
        match expr.node {
            ExprWhile(_, ref block, _) |
            ExprLoop(ref block, _, _) => {
                if never_loop(block, &expr.id) {
                    span_lint(cx, NEVER_LOOP, expr.span, "this loop never actually loops");
                }
            },
            _ => (),
        }

        // check for `loop { if let {} else break }` that could be `while let`
        // (also matches an explicit "match" instead of "if let")
        // (even if the "match" or "if let" is used for declaration)
        if let ExprLoop(ref block, _, LoopSource::Loop) = expr.node {
            // also check for empty `loop {}` statements
            if block.stmts.is_empty() && block.expr.is_none() {
                span_lint(
                    cx,
                    EMPTY_LOOP,
                    expr.span,
                    "empty `loop {}` detected. You may want to either use `panic!()` or add \
                     `std::thread::sleep(..);` to the loop body.",
                );
            }

            // extract the expression from the first statement (if any) in a block
            let inner_stmt_expr = extract_expr_from_first_stmt(block);
            // or extract the first expression (if any) from the block
            if let Some(inner) = inner_stmt_expr.or_else(|| extract_first_expr(block)) {
                if let ExprMatch(ref matchexpr, ref arms, ref source) = inner.node {
                    // ensure "if let" compatible match structure
                    match *source {
                        MatchSource::Normal |
                        MatchSource::IfLetDesugar { .. } => {
                            if arms.len() == 2 && arms[0].pats.len() == 1 && arms[0].guard.is_none() &&
                                arms[1].pats.len() == 1 && arms[1].guard.is_none() &&
                                is_simple_break_expr(&arms[1].body)
                            {
                                if in_external_macro(cx, expr.span) {
                                    return;
                                }

                                // NOTE: we used to make build a body here instead of using
                                // ellipsis, this was removed because:
                                // 1) it was ugly with big bodies;
                                // 2) it was not indented properly;
                                // 3) it wasn’t very smart (see #675).
                                span_lint_and_sugg(
                                    cx,
                                    WHILE_LET_LOOP,
                                    expr.span,
                                    "this loop could be written as a `while let` loop",
                                    "try",
                                    format!(
                                        "while let {} = {} {{ .. }}",
                                        snippet(cx, arms[0].pats[0].span, ".."),
                                        snippet(cx, matchexpr.span, "..")
                                    ),
                                );
                            }
                        },
                        _ => (),
                    }
                }
            }
        }
        if let ExprMatch(ref match_expr, ref arms, MatchSource::WhileLetDesugar) = expr.node {
            let pat = &arms[0].pats[0].node;
            if let (&PatKind::TupleStruct(ref qpath, ref pat_args, _),
                    &ExprMethodCall(ref method_path, _, ref method_args)) = (pat, &match_expr.node)
            {
                let iter_expr = &method_args[0];
                let lhs_constructor = last_path_segment(qpath);
                if method_path.name == "next" && match_trait_method(cx, match_expr, &paths::ITERATOR) &&
                    lhs_constructor.name == "Some" && !is_refutable(cx, &pat_args[0]) &&
                    !is_iterator_used_after_while_let(cx, iter_expr) &&
                    !is_nested(cx, expr, &method_args[0])
                {
                    let iterator = snippet(cx, method_args[0].span, "_");
                    let loop_var = snippet(cx, pat_args[0].span, "_");
                    span_lint_and_sugg(
                        cx,
                        WHILE_LET_ON_ITERATOR,
                        expr.span,
                        "this loop could be written as a `for` loop",
                        "try",
                        format!("for {} in {} {{ .. }}", loop_var, iterator),
                    );
                }
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt) {
        if let StmtSemi(ref expr, _) = stmt.node {
            if let ExprMethodCall(ref method, _, ref args) = expr.node {
                if args.len() == 1 && method.name == "collect" && match_trait_method(cx, expr, &paths::ITERATOR) {
                    span_lint(
                        cx,
                        UNUSED_COLLECT,
                        expr.span,
                        "you are collect()ing an iterator and throwing away the result. \
                         Consider using an explicit for loop to exhaust the iterator",
                    );
                }
            }
        }
    }
}

fn never_loop(block: &Block, id: &NodeId) -> bool {
    !contains_continue_block(block, id) && loop_exit_block(block)
}

fn contains_continue_block(block: &Block, dest: &NodeId) -> bool {
    block.stmts.iter().any(|e| contains_continue_stmt(e, dest)) ||
        block.expr.as_ref().map_or(
            false,
            |e| contains_continue_expr(e, dest),
        )
}

fn contains_continue_stmt(stmt: &Stmt, dest: &NodeId) -> bool {
    match stmt.node {
        StmtSemi(ref e, _) |
        StmtExpr(ref e, _) => contains_continue_expr(e, dest),
        StmtDecl(ref d, _) => contains_continue_decl(d, dest),
    }
}

fn contains_continue_decl(decl: &Decl, dest: &NodeId) -> bool {
    match decl.node {
        DeclLocal(ref local) => {
            local.init.as_ref().map_or(
                false,
                |e| contains_continue_expr(e, dest),
            )
        },
        _ => false,
    }
}

fn contains_continue_expr(expr: &Expr, dest: &NodeId) -> bool {
    match expr.node {
        ExprRet(Some(ref e)) |
        ExprBox(ref e) |
        ExprUnary(_, ref e) |
        ExprCast(ref e, _) |
        ExprType(ref e, _) |
        ExprField(ref e, _) |
        ExprTupField(ref e, _) |
        ExprAddrOf(_, ref e) |
        ExprRepeat(ref e, _) => contains_continue_expr(e, dest),
        ExprArray(ref es) |
        ExprMethodCall(_, _, ref es) |
        ExprTup(ref es) => es.iter().any(|e| contains_continue_expr(e, dest)),
        ExprCall(ref e, ref es) => {
            contains_continue_expr(e, dest) || es.iter().any(|e| contains_continue_expr(e, dest))
        },
        ExprBinary(_, ref e1, ref e2) |
        ExprAssign(ref e1, ref e2) |
        ExprAssignOp(_, ref e1, ref e2) |
        ExprIndex(ref e1, ref e2) => [e1, e2].iter().any(|e| contains_continue_expr(e, dest)),
        ExprIf(ref e, ref e2, ref e3) => {
            [e, e2].iter().chain(e3.as_ref().iter()).any(|e| {
                contains_continue_expr(e, dest)
            })
        },
        ExprWhile(ref e, ref b, _) => contains_continue_expr(e, dest) || contains_continue_block(b, dest),
        ExprMatch(ref e, ref arms, _) => {
            contains_continue_expr(e, dest) || arms.iter().any(|a| contains_continue_expr(&a.body, dest))
        },
        ExprBlock(ref block) |
        ExprLoop(ref block, ..) => contains_continue_block(block, dest),
        ExprStruct(_, _, ref base) => {
            base.as_ref().map_or(
                false,
                |e| contains_continue_expr(e, dest),
            )
        },
        ExprAgain(d) => d.target_id.opt_id().map_or(false, |id| id == *dest),
        _ => false,
    }
}

fn loop_exit_block(block: &Block) -> bool {
    block.stmts.iter().any(|e| loop_exit_stmt(e)) || block.expr.as_ref().map_or(false, |e| loop_exit_expr(e))
}

fn loop_exit_stmt(stmt: &Stmt) -> bool {
    match stmt.node {
        StmtSemi(ref e, _) |
        StmtExpr(ref e, _) => loop_exit_expr(e),
        StmtDecl(ref d, _) => loop_exit_decl(d),
    }
}

fn loop_exit_decl(decl: &Decl) -> bool {
    match decl.node {
        DeclLocal(ref local) => local.init.as_ref().map_or(false, |e| loop_exit_expr(e)),
        _ => false,
    }
}

fn loop_exit_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprBox(ref e) |
        ExprUnary(_, ref e) |
        ExprCast(ref e, _) |
        ExprType(ref e, _) |
        ExprField(ref e, _) |
        ExprTupField(ref e, _) |
        ExprAddrOf(_, ref e) |
        ExprRepeat(ref e, _) => loop_exit_expr(e),
        ExprArray(ref es) |
        ExprMethodCall(_, _, ref es) |
        ExprTup(ref es) => es.iter().any(|e| loop_exit_expr(e)),
        ExprCall(ref e, ref es) => loop_exit_expr(e) || es.iter().any(|e| loop_exit_expr(e)),
        ExprBinary(_, ref e1, ref e2) |
        ExprAssign(ref e1, ref e2) |
        ExprAssignOp(_, ref e1, ref e2) |
        ExprIndex(ref e1, ref e2) => [e1, e2].iter().any(|e| loop_exit_expr(e)),
        ExprIf(ref e, ref e2, ref e3) => {
            loop_exit_expr(e) || e3.as_ref().map_or(false, |e| loop_exit_expr(e)) && loop_exit_expr(e2)
        },
        ExprWhile(ref e, ref b, _) => loop_exit_expr(e) || loop_exit_block(b),
        ExprMatch(ref e, ref arms, _) => loop_exit_expr(e) || arms.iter().all(|a| loop_exit_expr(&a.body)),
        ExprBlock(ref b) => loop_exit_block(b),
        ExprBreak(_, _) | ExprAgain(_) | ExprRet(_) => true,
        _ => false,
    }
}

fn check_for_loop<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    pat: &'tcx Pat,
    arg: &'tcx Expr,
    body: &'tcx Expr,
    expr: &'tcx Expr,
) {
    check_for_loop_range(cx, pat, arg, body, expr);
    check_for_loop_reverse_range(cx, arg, expr);
    check_for_loop_arg(cx, pat, arg, expr);
    check_for_loop_explicit_counter(cx, arg, body, expr);
    check_for_loop_over_map_kv(cx, pat, arg, body, expr);
    check_for_mut_range_bound(cx, arg, body, expr);
    detect_manual_memcpy(cx, pat, arg, body, expr);
}

fn same_var<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &Expr, var: ast::NodeId) -> bool {
    if_let_chain! {[
        let ExprPath(ref qpath) = expr.node,
        let QPath::Resolved(None, ref path) = *qpath,
        path.segments.len() == 1,
        let Def::Local(local_id) = cx.tables.qpath_def(qpath, expr.hir_id),
        // our variable!
        local_id == var
    ], {
        return true;
    }}

    false
}

struct Offset {
    value: String,
    negate: bool,
}

impl Offset {
    fn negative(s: String) -> Self {
        Self {
            value: s,
            negate: true,
        }
    }

    fn positive(s: String) -> Self {
        Self {
            value: s,
            negate: false,
        }
    }
}

struct FixedOffsetVar {
    var_name: String,
    offset: Offset,
}

fn is_slice_like<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: Ty) -> bool {
    let is_slice = match ty.sty {
        ty::TyRef(_, ref subty) => is_slice_like(cx, subty.ty),
        ty::TySlice(..) | ty::TyArray(..) => true,
        _ => false,
    };

    is_slice || match_type(cx, ty, &paths::VEC) || match_type(cx, ty, &paths::VEC_DEQUE)
}

fn get_fixed_offset_var<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &Expr, var: ast::NodeId) -> Option<FixedOffsetVar> {
    fn extract_offset<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, e: &Expr, var: ast::NodeId) -> Option<String> {
        match e.node {
            ExprLit(ref l) => {
                match l.node {
                    ast::LitKind::Int(x, _ty) => Some(x.to_string()),
                    _ => None,
                }
            },
            ExprPath(..) if !same_var(cx, e, var) => Some(snippet_opt(cx, e.span).unwrap_or_else(|| "??".into())),
            _ => None,
        }
    }

    if let ExprIndex(ref seqexpr, ref idx) = expr.node {
        let ty = cx.tables.expr_ty(seqexpr);
        if !is_slice_like(cx, ty) {
            return None;
        }

        let offset = match idx.node {
            ExprBinary(op, ref lhs, ref rhs) => {
                match op.node {
                    BinOp_::BiAdd => {
                        let offset_opt = if same_var(cx, lhs, var) {
                            extract_offset(cx, rhs, var)
                        } else if same_var(cx, rhs, var) {
                            extract_offset(cx, lhs, var)
                        } else {
                            None
                        };

                        offset_opt.map(Offset::positive)
                    },
                    BinOp_::BiSub if same_var(cx, lhs, var) => extract_offset(cx, rhs, var).map(Offset::negative),
                    _ => None,
                }
            },
            ExprPath(..) => {
                if same_var(cx, idx, var) {
                    Some(Offset::positive("0".into()))
                } else {
                    None
                }
            },
            _ => None,
        };

        offset.map(|o| {
            FixedOffsetVar {
                var_name: snippet_opt(cx, seqexpr.span).unwrap_or_else(|| "???".into()),
                offset: o,
            }
        })
    } else {
        None
    }
}

fn fetch_cloned_fixed_offset_var<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    expr: &Expr,
    var: ast::NodeId,
) -> Option<FixedOffsetVar> {
    if_let_chain! {[
        let ExprMethodCall(ref method, _, ref args) = expr.node,
        method.name == "clone",
        args.len() == 1,
        let Some(arg) = args.get(0),
    ], {
        return get_fixed_offset_var(cx, arg, var);
    }}

    get_fixed_offset_var(cx, expr, var)
}

fn get_indexed_assignments<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    body: &Expr,
    var: ast::NodeId,
) -> Vec<(FixedOffsetVar, FixedOffsetVar)> {
    fn get_assignment<'a, 'tcx>(
        cx: &LateContext<'a, 'tcx>,
        e: &Expr,
        var: ast::NodeId,
    ) -> Option<(FixedOffsetVar, FixedOffsetVar)> {
        if let Expr_::ExprAssign(ref lhs, ref rhs) = e.node {
            match (get_fixed_offset_var(cx, lhs, var), fetch_cloned_fixed_offset_var(cx, rhs, var)) {
                (Some(offset_left), Some(offset_right)) => Some((offset_left, offset_right)),
                _ => None,
            }
        } else {
            None
        }
    }

    if let Expr_::ExprBlock(ref b) = body.node {
        let Block {
            ref stmts,
            ref expr,
            ..
        } = **b;

        stmts
            .iter()
            .map(|stmt| match stmt.node {
                Stmt_::StmtDecl(..) => None,
                Stmt_::StmtExpr(ref e, _node_id) |
                Stmt_::StmtSemi(ref e, _node_id) => Some(get_assignment(cx, e, var)),
            })
            .chain(expr.as_ref().into_iter().map(|e| {
                Some(get_assignment(cx, &*e, var))
            }))
            .filter_map(|op| op)
            .collect::<Option<Vec<_>>>()
            .unwrap_or_else(|| vec![])
    } else {
        get_assignment(cx, body, var).into_iter().collect()
    }
}

/// Check for for loops that sequentially copy items from one slice-like
/// object to another.
fn detect_manual_memcpy<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    pat: &'tcx Pat,
    arg: &'tcx Expr,
    body: &'tcx Expr,
    expr: &'tcx Expr,
) {
    if let Some(higher::Range {
                    start: Some(start),
                    ref end,
                    limits,
                }) = higher::range(arg)
    {
        // the var must be a single name
        if let PatKind::Binding(_, canonical_id, _, _) = pat.node {
            let print_sum = |arg1: &Offset, arg2: &Offset| -> String {
                match (&arg1.value[..], arg1.negate, &arg2.value[..], arg2.negate) {
                    ("0", _, "0", _) => "".into(),
                    ("0", _, x, false) |
                    (x, false, "0", false) => x.into(),
                    ("0", _, x, true) |
                    (x, false, "0", true) => format!("-{}", x),
                    (x, false, y, false) => format!("({} + {})", x, y),
                    (x, false, y, true) => format!("({} - {})", x, y),
                    (x, true, y, false) => format!("({} - {})", y, x),
                    (x, true, y, true) => format!("-({} + {})", x, y),
                }
            };

            let print_limit = |end: &Option<&Expr>, offset: Offset, var_name: &str| if let Some(end) = *end {
                if_let_chain! {[
                    let ExprMethodCall(ref method, _, ref len_args) = end.node,
                    method.name == "len",
                    len_args.len() == 1,
                    let Some(arg) = len_args.get(0),
                    snippet(cx, arg.span, "??") == var_name,
                ], {
                    return if offset.negate {
                        format!("({} - {})", snippet(cx, end.span, "<src>.len()"), offset.value)
                    } else {
                        "".to_owned()
                    };
                }}

                let end_str = match limits {
                    ast::RangeLimits::Closed => {
                        let end = sugg::Sugg::hir(cx, end, "<count>");
                        format!("{}", end + sugg::ONE)
                    },
                    ast::RangeLimits::HalfOpen => format!("{}", snippet(cx, end.span, "..")),
                };

                print_sum(&Offset::positive(end_str), &offset)
            } else {
                "..".into()
            };

            // The only statements in the for loops can be indexed assignments from
            // indexed retrievals.
            let manual_copies = get_indexed_assignments(cx, body, canonical_id);

            let big_sugg = manual_copies
                .into_iter()
                .map(|(dst_var, src_var)| {
                    let start_str = Offset::positive(snippet_opt(cx, start.span).unwrap_or_else(|| "".into()));
                    let dst_offset = print_sum(&start_str, &dst_var.offset);
                    let dst_limit = print_limit(end, dst_var.offset, &dst_var.var_name);
                    let src_offset = print_sum(&start_str, &src_var.offset);
                    let src_limit = print_limit(end, src_var.offset, &src_var.var_name);
                    let dst = if dst_offset == "" && dst_limit == "" {
                        dst_var.var_name
                    } else {
                        format!("{}[{}..{}]", dst_var.var_name, dst_offset, dst_limit)
                    };

                    format!("{}.clone_from_slice(&{}[{}..{}])", dst, src_var.var_name, src_offset, src_limit)
                })
                .join("\n    ");

            if !big_sugg.is_empty() {
                span_lint_and_sugg(
                    cx,
                    MANUAL_MEMCPY,
                    expr.span,
                    "it looks like you're manually copying between slices",
                    "try replacing the loop by",
                    big_sugg,
                );
            }
        }
    }
}

/// Check for looping over a range and then indexing a sequence with it.
/// The iteratee must be a range literal.
fn check_for_loop_range<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    pat: &'tcx Pat,
    arg: &'tcx Expr,
    body: &'tcx Expr,
    expr: &'tcx Expr,
) {
    if let Some(higher::Range {
                    start: Some(start),
                    ref end,
                    limits,
                }) = higher::range(arg)
    {
        // the var must be a single name
        if let PatKind::Binding(_, canonical_id, ref ident, _) = pat.node {
            let mut visitor = VarVisitor {
                cx: cx,
                var: canonical_id,
                indexed: HashMap::new(),
                referenced: HashSet::new(),
                nonindex: false,
            };
            walk_expr(&mut visitor, body);

            // linting condition: we only indexed one variable
            if visitor.indexed.len() == 1 {
                let (indexed, indexed_extent) = visitor.indexed.into_iter().next().expect(
                    "already checked that we have exactly 1 element",
                );

                // ensure that the indexed variable was declared before the loop, see #601
                if let Some(indexed_extent) = indexed_extent {
                    let parent_id = cx.tcx.hir.get_parent(expr.id);
                    let parent_def_id = cx.tcx.hir.local_def_id(parent_id);
                    let region_scope_tree = cx.tcx.region_scope_tree(parent_def_id);
                    let pat_extent = region_scope_tree.var_scope(pat.hir_id.local_id);
                    if region_scope_tree.is_subscope_of(indexed_extent, pat_extent) {
                        return;
                    }
                }

                // don't lint if the container that is indexed into is also used without
                // indexing
                if visitor.referenced.contains(&indexed) {
                    return;
                }

                let starts_at_zero = is_integer_literal(start, 0);

                let skip = if starts_at_zero {
                    "".to_owned()
                } else {
                    format!(".skip({})", snippet(cx, start.span, ".."))
                };

                let take = if let Some(end) = *end {
                    if is_len_call(end, &indexed) {
                        "".to_owned()
                    } else {
                        match limits {
                            ast::RangeLimits::Closed => {
                                let end = sugg::Sugg::hir(cx, end, "<count>");
                                format!(".take({})", end + sugg::ONE)
                            },
                            ast::RangeLimits::HalfOpen => format!(".take({})", snippet(cx, end.span, "..")),
                        }
                    }
                } else {
                    "".to_owned()
                };

                if visitor.nonindex {
                    span_lint_and_then(
                        cx,
                        NEEDLESS_RANGE_LOOP,
                        expr.span,
                        &format!("the loop variable `{}` is used to index `{}`", ident.node, indexed),
                        |db| {
                            multispan_sugg(
                                db,
                                "consider using an iterator".to_string(),
                                vec![
                                    (pat.span, format!("({}, <item>)", ident.node)),
                                    (arg.span, format!("{}.iter().enumerate(){}{}", indexed, take, skip)),
                                ],
                            );
                        },
                    );
                } else {
                    let repl = if starts_at_zero && take.is_empty() {
                        format!("&{}", indexed)
                    } else {
                        format!("{}.iter(){}{}", indexed, take, skip)
                    };

                    span_lint_and_then(
                        cx,
                        NEEDLESS_RANGE_LOOP,
                        expr.span,
                        &format!("the loop variable `{}` is only used to index `{}`.", ident.node, indexed),
                        |db| {
                            multispan_sugg(
                                db,
                                "consider using an iterator".to_string(),
                                vec![(pat.span, "<item>".to_string()), (arg.span, repl)],
                            );
                        },
                    );
                }
            }
        }
    }
}

fn is_len_call(expr: &Expr, var: &Name) -> bool {
    if_let_chain! {[
        let ExprMethodCall(ref method, _, ref len_args) = expr.node,
        len_args.len() == 1,
        method.name == "len",
        let ExprPath(QPath::Resolved(_, ref path)) = len_args[0].node,
        path.segments.len() == 1,
        path.segments[0].name == *var
    ], {
        return true;
    }}

    false
}

fn check_for_loop_reverse_range<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, arg: &'tcx Expr, expr: &'tcx Expr) {
    // if this for loop is iterating over a two-sided range...
    if let Some(higher::Range {
                    start: Some(start),
                    end: Some(end),
                    limits,
                }) = higher::range(arg)
    {
        // ...and both sides are compile-time constant integers...
        let parent_item = cx.tcx.hir.get_parent(arg.id);
        let parent_def_id = cx.tcx.hir.local_def_id(parent_item);
        let substs = Substs::identity_for_item(cx.tcx, parent_def_id);
        let constcx = ConstContext::new(cx.tcx, cx.param_env.and(substs), cx.tables);
        if let Ok(start_idx) = constcx.eval(start) {
            if let Ok(end_idx) = constcx.eval(end) {
                // ...and the start index is greater than the end index,
                // this loop will never run. This is often confusing for developers
                // who think that this will iterate from the larger value to the
                // smaller value.
                let (sup, eq) = match (start_idx, end_idx) {
                    (&ty::Const { val: ConstVal::Integral(start_idx), .. },
                     &ty::Const { val: ConstVal::Integral(end_idx), .. }) => {
                        (start_idx > end_idx, start_idx == end_idx)
                    },
                    _ => (false, false),
                };

                if sup {
                    let start_snippet = snippet(cx, start.span, "_");
                    let end_snippet = snippet(cx, end.span, "_");
                    let dots = if limits == ast::RangeLimits::Closed {
                        "..."
                    } else {
                        ".."
                    };

                    span_lint_and_then(
                        cx,
                        REVERSE_RANGE_LOOP,
                        expr.span,
                        "this range is empty so this for loop will never run",
                        |db| {
                            db.span_suggestion(
                                arg.span,
                                "consider using the following if you are attempting to iterate over this \
                                 range in reverse",
                                format!(
                                    "({end}{dots}{start}).rev()",
                                    end = end_snippet,
                                    dots = dots,
                                    start = start_snippet
                                ),
                            );
                        },
                    );
                } else if eq && limits != ast::RangeLimits::Closed {
                    // if they are equal, it's also problematic - this loop
                    // will never run.
                    span_lint(
                        cx,
                        REVERSE_RANGE_LOOP,
                        expr.span,
                        "this range is empty so this for loop will never run",
                    );
                }
            }
        }
    }
}

fn lint_iter_method(cx: &LateContext, args: &[Expr], arg: &Expr, method_name: &str) {
    let object = snippet(cx, args[0].span, "_");
    let muta = if method_name == "iter_mut" {
        "mut "
    } else {
        ""
    };
    span_lint_and_sugg(
        cx,
        EXPLICIT_ITER_LOOP,
        arg.span,
        "it is more idiomatic to loop over references to containers instead of using explicit \
         iteration methods",
        "to write this more concisely, try",
        format!("&{}{}", muta, object),
    )
}

fn check_for_loop_arg(cx: &LateContext, pat: &Pat, arg: &Expr, expr: &Expr) {
    let mut next_loop_linted = false; // whether or not ITER_NEXT_LOOP lint was used
    if let ExprMethodCall(ref method, _, ref args) = arg.node {
        // just the receiver, no arguments
        if args.len() == 1 {
            let method_name = &*method.name.as_str();
            // check for looping over x.iter() or x.iter_mut(), could use &x or &mut x
            if method_name == "iter" || method_name == "iter_mut" {
                if is_ref_iterable_type(cx, &args[0]) {
                    lint_iter_method(cx, args, arg, method_name);
                }
            } else if method_name == "into_iter" && match_trait_method(cx, arg, &paths::INTO_ITERATOR) {
                let def_id = cx.tables.type_dependent_defs()[arg.hir_id].def_id();
                let substs = cx.tables.node_substs(arg.hir_id);
                let method_type = cx.tcx.type_of(def_id).subst(cx.tcx, substs);

                let fn_arg_tys = method_type.fn_sig(cx.tcx).inputs();
                assert_eq!(fn_arg_tys.skip_binder().len(), 1);
                if fn_arg_tys.skip_binder()[0].is_region_ptr() {
                    lint_iter_method(cx, args, arg, method_name);
                } else {
                    let object = snippet(cx, args[0].span, "_");
                    span_lint_and_sugg(
                        cx,
                        EXPLICIT_INTO_ITER_LOOP,
                        arg.span,
                        "it is more idiomatic to loop over containers instead of using explicit \
                         iteration methods`",
                        "to write this more concisely, try",
                        object.to_string(),
                    );
                }
            } else if method_name == "next" && match_trait_method(cx, arg, &paths::ITERATOR) {
                span_lint(
                    cx,
                    ITER_NEXT_LOOP,
                    expr.span,
                    "you are iterating over `Iterator::next()` which is an Option; this will compile but is \
                     probably not what you want",
                );
                next_loop_linted = true;
            }
        }
    }
    if !next_loop_linted {
        check_arg_type(cx, pat, arg);
    }
}

/// Check for `for` loops over `Option`s and `Results`
fn check_arg_type(cx: &LateContext, pat: &Pat, arg: &Expr) {
    let ty = cx.tables.expr_ty(arg);
    if match_type(cx, ty, &paths::OPTION) {
        span_help_and_lint(
            cx,
            FOR_LOOP_OVER_OPTION,
            arg.span,
            &format!(
                "for loop over `{0}`, which is an `Option`. This is more readably written as an \
                 `if let` statement.",
                snippet(cx, arg.span, "_")
            ),
            &format!(
                "consider replacing `for {0} in {1}` with `if let Some({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            ),
        );
    } else if match_type(cx, ty, &paths::RESULT) {
        span_help_and_lint(
            cx,
            FOR_LOOP_OVER_RESULT,
            arg.span,
            &format!(
                "for loop over `{0}`, which is a `Result`. This is more readably written as an \
                 `if let` statement.",
                snippet(cx, arg.span, "_")
            ),
            &format!(
                "consider replacing `for {0} in {1}` with `if let Ok({0}) = {1}`",
                snippet(cx, pat.span, "_"),
                snippet(cx, arg.span, "_")
            ),
        );
    }
}

fn check_for_loop_explicit_counter<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    arg: &'tcx Expr,
    body: &'tcx Expr,
    expr: &'tcx Expr,
) {
    // Look for variables that are incremented once per loop iteration.
    let mut visitor = IncrementVisitor {
        cx: cx,
        states: HashMap::new(),
        depth: 0,
        done: false,
    };
    walk_expr(&mut visitor, body);

    // For each candidate, check the parent block to see if
    // it's initialized to zero at the start of the loop.
    let map = &cx.tcx.hir;
    let parent_scope = map.get_enclosing_scope(expr.id).and_then(|id| {
        map.get_enclosing_scope(id)
    });
    if let Some(parent_id) = parent_scope {
        if let NodeBlock(block) = map.get(parent_id) {
            for (id, _) in visitor.states.iter().filter(
                |&(_, v)| *v == VarState::IncrOnce,
            )
            {
                let mut visitor2 = InitializeVisitor {
                    cx: cx,
                    end_expr: expr,
                    var_id: *id,
                    state: VarState::IncrOnce,
                    name: None,
                    depth: 0,
                    past_loop: false,
                };
                walk_block(&mut visitor2, block);

                if visitor2.state == VarState::Warn {
                    if let Some(name) = visitor2.name {
                        span_lint(
                            cx,
                            EXPLICIT_COUNTER_LOOP,
                            expr.span,
                            &format!(
                                "the variable `{0}` is used as a loop counter. Consider using `for ({0}, \
                                 item) in {1}.enumerate()` or similar iterators",
                                name,
                                snippet(cx, arg.span, "_")
                            ),
                        );
                    }
                }
            }
        }
    }
}

/// Check for the `FOR_KV_MAP` lint.
fn check_for_loop_over_map_kv<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    pat: &'tcx Pat,
    arg: &'tcx Expr,
    body: &'tcx Expr,
    expr: &'tcx Expr,
) {
    let pat_span = pat.span;

    if let PatKind::Tuple(ref pat, _) = pat.node {
        if pat.len() == 2 {
            let arg_span = arg.span;
            let (new_pat_span, kind, ty, mutbl) = match cx.tables.expr_ty(arg).sty {
                ty::TyRef(_, ref tam) => {
                    match (&pat[0].node, &pat[1].node) {
                        (key, _) if pat_is_wild(key, body) => (pat[1].span, "value", tam.ty, tam.mutbl),
                        (_, value) if pat_is_wild(value, body) => (pat[0].span, "key", tam.ty, MutImmutable),
                        _ => return,
                    }
                },
                _ => return,
            };
            let mutbl = match mutbl {
                MutImmutable => "",
                MutMutable => "_mut",
            };
            let arg = match arg.node {
                ExprAddrOf(_, ref expr) => &**expr,
                _ => arg,
            };

            if match_type(cx, ty, &paths::HASHMAP) || match_type(cx, ty, &paths::BTREEMAP) {
                span_lint_and_then(
                    cx,
                    FOR_KV_MAP,
                    expr.span,
                    &format!("you seem to want to iterate on a map's {}s", kind),
                    |db| {
                        let map = sugg::Sugg::hir(cx, arg, "map");
                        multispan_sugg(
                            db,
                            "use the corresponding method".into(),
                            vec![
                                (pat_span, snippet(cx, new_pat_span, kind).into_owned()),
                                (arg_span, format!("{}.{}s{}()", map.maybe_par(), kind, mutbl)),
                            ],
                        );
                    },
                );
            }
        }
    }
}

// TODO: clippy builds, but the `mutate` method of `Delegate` is never called when compiling `tests/run-pass/mut_range_bound_tmp.rs`. what's wrong? 

struct MutateDelegate<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    node_id: NodeId,
    was_mutated: bool
}

impl<'a, 'tcx> Delegate<'tcx> for MutateDelegate<'a, 'tcx> {
    fn consume(&mut self, _: NodeId, _: Span, cmt: cmt<'tcx>, mode: ConsumeMode) {
    }
  
    fn matched_pat(&mut self, matched_pat: &Pat, cmt: cmt<'tcx>, mode: MatchMode) {
    }

    fn consume_pat(&mut self, consume_pat: &Pat, cmt: cmt<'tcx>, mode: ConsumeMode) {
    }

    fn borrow(&mut self, _: NodeId, _: Span, _: cmt<'tcx>, _: ty::Region, _: ty::BorrowKind, _: LoanCause) {        
    }

    fn mutate(&mut self, assignment_id: NodeId, sp: Span, _: cmt<'tcx>, _: MutateMode) {
        self.cx.sess().span_note_without_error(sp, "mutates!");
        if assignment_id == self.node_id {
            self.was_mutated = true;
        }
    }

    fn decl_without_init(&mut self, _: NodeId, _: Span) {
    }
}

impl<'a, 'tcx> MutateDelegate<'a, 'tcx> {
    fn bound_was_mutated(&self) -> bool {
        self.was_mutated
    }
}

fn check_for_mut_range_bound(cx: &LateContext, arg: &Expr, body: &Expr,  expr: &Expr) {
    if let Some(higher::Range { start: Some(start), end: Some(end), limits }) = higher::range(arg) {
        let bounds = vec![start, end];
        for bound in &bounds {
            if check_for_mutation(cx, body, bound) {
                span_lint(cx, MUT_RANGE_BOUND, expr.span, "you are looping over a range where at least one bound was defined as a mutable variable. keep in mind that mutating this variable inside the loop will not affect the range");   
                return;
            }
        }
    }
}

fn check_for_mutation(cx: &LateContext, body: &Expr, bound: &Expr) -> bool {
    if_let_chain! {[
        let ExprPath(ref qpath) = bound.node,
        let QPath::Resolved(None, ref path) = *qpath,
    ], {
        let def = cx.tables.qpath_def(qpath, bound.hir_id);

                    cx.sess().span_note_without_error(body.span, "loop");
        match def {
            Def::Local(..) | Def::Upvar(..) => {
                let def_id = def.def_id();
                let node_id = cx.tcx.hir.as_local_node_id(def_id).expect("local/upvar are local nodes");
                let node_str = cx.tcx.hir.get(node_id);
                if_let_chain! {[
                    let map::Node::NodeBinding(pat) = node_str,
                    let PatKind::Binding(bind_ann, _, _, _) = pat.node,
                    let BindingAnnotation::Mutable = bind_ann,
                    
                ], {
                    let mut delegate = MutateDelegate { cx: cx, node_id: node_id, was_mutated: false };
                    let region_maps = &cx.tcx.region_maps(def_id); // is this the correct argument?
                    ExprUseVisitor::new(&mut delegate, cx.tcx, cx.param_env, region_maps, cx.tables).walk_expr(body);
                    return delegate.bound_was_mutated();
                }}
            },
            _ => (),
        }}
    }
    return false;
}

/// Return true if the pattern is a `PatWild` or an ident prefixed with `'_'`.
fn pat_is_wild<'tcx>(pat: &'tcx PatKind, body: &'tcx Expr) -> bool {
    match *pat {
        PatKind::Wild => true,
        PatKind::Binding(_, _, ident, None) if ident.node.as_str().starts_with('_') => {
            let mut visitor = UsedVisitor {
                var: ident.node,
                used: false,
            };
            walk_expr(&mut visitor, body);
            !visitor.used
        },
        _ => false,
    }
}

struct UsedVisitor {
    var: ast::Name, // var to look for
    used: bool, // has the var been used otherwise?
}

impl<'tcx> Visitor<'tcx> for UsedVisitor {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if match_var(expr, self.var) {
            self.used = true;
        } else {
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

struct LocalUsedVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    local: ast::NodeId,
    used: bool,
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for LocalUsedVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if same_var(self.cx, expr, self.local) {
            self.used = true;
        } else {
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

struct VarVisitor<'a, 'tcx: 'a> {
    /// context reference
    cx: &'a LateContext<'a, 'tcx>,
    /// var name to look for as index
    var: ast::NodeId,
    /// indexed variables, the extend is `None` for global
    indexed: HashMap<Name, Option<region::Scope>>,
    /// Any names that are used outside an index operation.
    /// Used to detect things like `&mut vec` used together with `vec[i]`
    referenced: HashSet<Name>,
    /// has the loop variable been used in expressions other than the index of
    /// an index op?
    nonindex: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for VarVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if_let_chain! {[
            // an index op
            let ExprIndex(ref seqexpr, ref idx) = expr.node,
            // the indexed container is referenced by a name
            let ExprPath(ref seqpath) = seqexpr.node,
            let QPath::Resolved(None, ref seqvar) = *seqpath,
            seqvar.segments.len() == 1,
        ], {
            let index_used = same_var(self.cx, idx, self.var) || {
                let mut used_visitor = LocalUsedVisitor {
                    cx: self.cx,
                    local: self.var,
                    used: false,
                };
                walk_expr(&mut used_visitor, idx);
                used_visitor.used
            };

            if index_used {
                let def = self.cx.tables.qpath_def(seqpath, seqexpr.hir_id);
                match def {
                    Def::Local(node_id) | Def::Upvar(node_id, ..) => {
                        let hir_id = self.cx.tcx.hir.node_to_hir_id(node_id);

                        let parent_id = self.cx.tcx.hir.get_parent(expr.id);
                        let parent_def_id = self.cx.tcx.hir.local_def_id(parent_id);
                        let extent = self.cx.tcx.region_scope_tree(parent_def_id).var_scope(hir_id.local_id);
                        self.indexed.insert(seqvar.segments[0].name, Some(extent));
                        return;  // no need to walk further *on the variable*
                    }
                    Def::Static(..) | Def::Const(..) => {
                        self.indexed.insert(seqvar.segments[0].name, None);
                        return;  // no need to walk further *on the variable*
                    }
                    _ => (),
                }
            }
        }}

        if_let_chain! {[
            // directly using a variable
            let ExprPath(ref qpath) = expr.node,
            let QPath::Resolved(None, ref path) = *qpath,
            path.segments.len() == 1,
            let Def::Local(local_id) = self.cx.tables.qpath_def(qpath, expr.hir_id),
        ], {
            if local_id == self.var {
                // we are not indexing anything, record that
                self.nonindex = true;
            } else {
                // not the correct variable, but still a variable
                self.referenced.insert(path.segments[0].name);
            }
        }}

        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

fn is_iterator_used_after_while_let<'a, 'tcx: 'a>(cx: &LateContext<'a, 'tcx>, iter_expr: &'tcx Expr) -> bool {
    let def_id = match var_def_id(cx, iter_expr) {
        Some(id) => id,
        None => return false,
    };
    let mut visitor = VarUsedAfterLoopVisitor {
        cx: cx,
        def_id: def_id,
        iter_expr_id: iter_expr.id,
        past_while_let: false,
        var_used_after_while_let: false,
    };
    if let Some(enclosing_block) = get_enclosing_block(cx, def_id) {
        walk_block(&mut visitor, enclosing_block);
    }
    visitor.var_used_after_while_let
}

struct VarUsedAfterLoopVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    def_id: NodeId,
    iter_expr_id: NodeId,
    past_while_let: bool,
    var_used_after_while_let: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for VarUsedAfterLoopVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if self.past_while_let {
            if Some(self.def_id) == var_def_id(self.cx, expr) {
                self.var_used_after_while_let = true;
            }
        } else if self.iter_expr_id == expr.id {
            self.past_while_let = true;
        }
        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}


/// Return true if the type of expr is one that provides `IntoIterator` impls
/// for `&T` and `&mut T`, such as `Vec`.
#[cfg_attr(rustfmt, rustfmt_skip)]
fn is_ref_iterable_type(cx: &LateContext, e: &Expr) -> bool {
    // no walk_ptrs_ty: calling iter() on a reference can make sense because it
    // will allow further borrows afterwards
    let ty = cx.tables.expr_ty(e);
    is_iterable_array(ty) ||
    match_type(cx, ty, &paths::VEC) ||
    match_type(cx, ty, &paths::LINKED_LIST) ||
    match_type(cx, ty, &paths::HASHMAP) ||
    match_type(cx, ty, &paths::HASHSET) ||
    match_type(cx, ty, &paths::VEC_DEQUE) ||
    match_type(cx, ty, &paths::BINARY_HEAP) ||
    match_type(cx, ty, &paths::BTREEMAP) ||
    match_type(cx, ty, &paths::BTREESET)
}

fn is_iterable_array(ty: Ty) -> bool {
    // IntoIterator is currently only implemented for array sizes <= 32 in rustc
    match ty.sty {
        ty::TyArray(_, n) => (0...32).contains(const_to_u64(n)),
        _ => false,
    }
}

/// If a block begins with a statement (possibly a `let` binding) and has an
/// expression, return it.
fn extract_expr_from_first_stmt(block: &Block) -> Option<&Expr> {
    if block.stmts.is_empty() {
        return None;
    }
    if let StmtDecl(ref decl, _) = block.stmts[0].node {
        if let DeclLocal(ref local) = decl.node {
            if let Some(ref expr) = local.init {
                Some(expr)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

/// If a block begins with an expression (with or without semicolon), return it.
fn extract_first_expr(block: &Block) -> Option<&Expr> {
    match block.expr {
        Some(ref expr) if block.stmts.is_empty() => Some(expr),
        None if !block.stmts.is_empty() => {
            match block.stmts[0].node {
                StmtExpr(ref expr, _) |
                StmtSemi(ref expr, _) => Some(expr),
                StmtDecl(..) => None,
            }
        },
        _ => None,
    }
}

/// Return true if expr contains a single break expr without destination label
/// and
/// passed expression. The expression may be within a block.
fn is_simple_break_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprBreak(dest, ref passed_expr) if dest.ident.is_none() && passed_expr.is_none() => true,
        ExprBlock(ref b) => {
            match extract_first_expr(b) {
                Some(subexpr) => is_simple_break_expr(subexpr),
                None => false,
            }
        },
        _ => false,
    }
}

// To trigger the EXPLICIT_COUNTER_LOOP lint, a variable must be
// incremented exactly once in the loop body, and initialized to zero
// at the start of the loop.
#[derive(PartialEq)]
enum VarState {
    Initial, // Not examined yet
    IncrOnce, // Incremented exactly once, may be a loop counter
    Declared, // Declared but not (yet) initialized to zero
    Warn,
    DontWarn,
}

/// Scan a for loop for variables that are incremented exactly once.
struct IncrementVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>, // context reference
    states: HashMap<NodeId, VarState>, // incremented variables
    depth: u32, // depth of conditional expressions
    done: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for IncrementVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if self.done {
            return;
        }

        // If node is a variable
        if let Some(def_id) = var_def_id(self.cx, expr) {
            if let Some(parent) = get_parent_expr(self.cx, expr) {
                let state = self.states.entry(def_id).or_insert(VarState::Initial);

                match parent.node {
                    ExprAssignOp(op, ref lhs, ref rhs) => {
                        if lhs.id == expr.id {
                            if op.node == BiAdd && is_integer_literal(rhs, 1) {
                                *state = match *state {
                                    VarState::Initial if self.depth == 0 => VarState::IncrOnce,
                                    _ => VarState::DontWarn,
                                };
                            } else {
                                // Assigned some other value
                                *state = VarState::DontWarn;
                            }
                        }
                    },
                    ExprAssign(ref lhs, _) if lhs.id == expr.id => *state = VarState::DontWarn,
                    ExprAddrOf(mutability, _) if mutability == MutMutable => *state = VarState::DontWarn,
                    _ => (),
                }
            }
        } else if is_loop(expr) {
            self.states.clear();
            self.done = true;
            return;
        } else if is_conditional(expr) {
            self.depth += 1;
            walk_expr(self, expr);
            self.depth -= 1;
            return;
        }
        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

/// Check whether a variable is initialized to zero at the start of a loop.
struct InitializeVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>, // context reference
    end_expr: &'tcx Expr, // the for loop. Stop scanning here.
    var_id: NodeId,
    state: VarState,
    name: Option<Name>,
    depth: u32, // depth of conditional expressions
    past_loop: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for InitializeVisitor<'a, 'tcx> {
    fn visit_decl(&mut self, decl: &'tcx Decl) {
        // Look for declarations of the variable
        if let DeclLocal(ref local) = decl.node {
            if local.pat.id == self.var_id {
                if let PatKind::Binding(_, _, ref ident, _) = local.pat.node {
                    self.name = Some(ident.node);

                    self.state = if let Some(ref init) = local.init {
                        if is_integer_literal(init, 0) {
                            VarState::Warn
                        } else {
                            VarState::Declared
                        }
                    } else {
                        VarState::Declared
                    }
                }
            }
        }
        walk_decl(self, decl);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if self.state == VarState::DontWarn {
            return;
        }
        if expr == self.end_expr {
            self.past_loop = true;
            return;
        }
        // No need to visit expressions before the variable is
        // declared
        if self.state == VarState::IncrOnce {
            return;
        }

        // If node is the desired variable, see how it's used
        if var_def_id(self.cx, expr) == Some(self.var_id) {
            if let Some(parent) = get_parent_expr(self.cx, expr) {
                match parent.node {
                    ExprAssignOp(_, ref lhs, _) if lhs.id == expr.id => {
                        self.state = VarState::DontWarn;
                    },
                    ExprAssign(ref lhs, ref rhs) if lhs.id == expr.id => {
                        self.state = if is_integer_literal(rhs, 0) && self.depth == 0 {
                            VarState::Warn
                        } else {
                            VarState::DontWarn
                        }
                    },
                    ExprAddrOf(mutability, _) if mutability == MutMutable => self.state = VarState::DontWarn,
                    _ => (),
                }
            }

            if self.past_loop {
                self.state = VarState::DontWarn;
                return;
            }
        } else if !self.past_loop && is_loop(expr) {
            self.state = VarState::DontWarn;
            return;
        } else if is_conditional(expr) {
            self.depth += 1;
            walk_expr(self, expr);
            self.depth -= 1;
            return;
        }
        walk_expr(self, expr);
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

fn var_def_id(cx: &LateContext, expr: &Expr) -> Option<NodeId> {
    if let ExprPath(ref qpath) = expr.node {
        let path_res = cx.tables.qpath_def(qpath, expr.hir_id);
        if let Def::Local(node_id) = path_res {
            return Some(node_id);
        }
    }
    None
}

fn is_loop(expr: &Expr) -> bool {
    match expr.node {
        ExprLoop(..) | ExprWhile(..) => true,
        _ => false,
    }
}

fn is_conditional(expr: &Expr) -> bool {
    match expr.node {
        ExprIf(..) | ExprMatch(..) => true,
        _ => false,
    }
}

fn is_nested(cx: &LateContext, match_expr: &Expr, iter_expr: &Expr) -> bool {
    if_let_chain! {[
        let Some(loop_block) = get_enclosing_block(cx, match_expr.id),
        let Some(map::Node::NodeExpr(loop_expr)) = cx.tcx.hir.find(cx.tcx.hir.get_parent_node(loop_block.id)),
    ], {
        return is_loop_nested(cx, loop_expr, iter_expr)
    }}
    false
}

fn is_loop_nested(cx: &LateContext, loop_expr: &Expr, iter_expr: &Expr) -> bool {
    let mut id = loop_expr.id;
    let iter_name = if let Some(name) = path_name(iter_expr) {
        name
    } else {
        return true;
    };
    loop {
        let parent = cx.tcx.hir.get_parent_node(id);
        if parent == id {
            return false;
        }
        match cx.tcx.hir.find(parent) {
            Some(NodeExpr(expr)) => {
                match expr.node {
                    ExprLoop(..) | ExprWhile(..) => {
                        return true;
                    },
                    _ => (),
                }
            },
            Some(NodeBlock(block)) => {
                let mut block_visitor = LoopNestVisitor {
                    id: id,
                    iterator: iter_name,
                    nesting: Unknown,
                };
                walk_block(&mut block_visitor, block);
                if block_visitor.nesting == RuledOut {
                    return false;
                }
            },
            Some(NodeStmt(_)) => (),
            _ => {
                return false;
            },
        }
        id = parent;
    }
}

#[derive(PartialEq, Eq)]
enum Nesting {
    Unknown, // no nesting detected yet
    RuledOut, // the iterator is initialized or assigned within scope
    LookFurther, // no nesting detected, no further walk required
}

use self::Nesting::{LookFurther, RuledOut, Unknown};

struct LoopNestVisitor {
    id: NodeId,
    iterator: Name,
    nesting: Nesting,
}

impl<'tcx> Visitor<'tcx> for LoopNestVisitor {
    fn visit_stmt(&mut self, stmt: &'tcx Stmt) {
        if stmt.node.id() == self.id {
            self.nesting = LookFurther;
        } else if self.nesting == Unknown {
            walk_stmt(self, stmt);
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        if self.nesting != Unknown {
            return;
        }
        if expr.id == self.id {
            self.nesting = LookFurther;
            return;
        }
        match expr.node {
            ExprAssign(ref path, _) |
            ExprAssignOp(_, ref path, _) => {
                if match_var(path, self.iterator) {
                    self.nesting = RuledOut;
                }
            },
            _ => walk_expr(self, expr),
        }
    }

    fn visit_pat(&mut self, pat: &'tcx Pat) {
        if self.nesting != Unknown {
            return;
        }
        if let PatKind::Binding(_, _, span_name, _) = pat.node {
            if self.iterator == span_name.node {
                self.nesting = RuledOut;
                return;
            }
        }
        walk_pat(self, pat)
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

fn path_name(e: &Expr) -> Option<Name> {
    if let ExprPath(QPath::Resolved(_, ref path)) = e.node {
        let segments = &path.segments;
        if segments.len() == 1 {
            return Some(segments[0].name);
        }
    };
    None
}

use rustc::lint::*;
use rustc_front::hir::*;
use reexport::*;
use rustc_front::visit::{Visitor, walk_expr, walk_block, walk_decl};
use rustc::middle::ty;
use rustc::middle::def::DefLocal;
use consts::{constant_simple, Constant};
use rustc::front::map::Node::{NodeBlock};
use std::collections::{HashSet,HashMap};

use utils::{snippet, span_lint, get_parent_expr, match_trait_method, match_type,
            in_external_macro, expr_block, span_help_and_lint, is_integer_literal};
use utils::{VEC_PATH, LL_PATH};

declare_lint!{ pub NEEDLESS_RANGE_LOOP, Warn,
               "for-looping over a range of indices where an iterator over items would do" }

declare_lint!{ pub EXPLICIT_ITER_LOOP, Warn,
               "for-looping over `_.iter()` or `_.iter_mut()` when `&_` or `&mut _` would do" }

declare_lint!{ pub ITER_NEXT_LOOP, Warn,
               "for-looping over `_.next()` which is probably not intended" }

declare_lint!{ pub WHILE_LET_LOOP, Warn,
               "`loop { if let { ... } else break }` can be written as a `while let` loop" }

declare_lint!{ pub UNUSED_COLLECT, Warn,
               "`collect()`ing an iterator without using the result; this is usually better \
                written as a for loop" }

declare_lint!{ pub REVERSE_RANGE_LOOP, Warn,
               "Iterating over an empty range, such as `10..0` or `5..5`" }

declare_lint!{ pub EXPLICIT_COUNTER_LOOP, Warn,
               "for-looping with an explicit counter when `_.enumerate()` would do" }

#[derive(Copy, Clone)]
pub struct LoopsPass;

impl LintPass for LoopsPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_RANGE_LOOP, EXPLICIT_ITER_LOOP, ITER_NEXT_LOOP,
                    WHILE_LET_LOOP, UNUSED_COLLECT, REVERSE_RANGE_LOOP, EXPLICIT_COUNTER_LOOP)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let Some((pat, arg, body)) = recover_for_loop(expr) {
            // check for looping over a range and then indexing a sequence with it
            // -> the iteratee must be a range literal
            if let ExprRange(Some(ref l), _) = arg.node {
                // Range should start with `0`
                if let ExprLit(ref lit) = l.node {
                    if let LitInt(0, _) = lit.node {

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
                }
            }

            // if this for loop is iterating over a two-sided range...
            if let ExprRange(Some(ref start_expr), Some(ref stop_expr)) = arg.node {
                // ...and both sides are compile-time constant integers...
                if let Some(Constant::ConstantInt(start_idx, _)) = constant_simple(start_expr) {
                    if let Some(Constant::ConstantInt(stop_idx, _)) = constant_simple(stop_expr) {
                        // ...and the start index is greater than the stop index,
                        // this loop will never run. This is often confusing for developers
                        // who think that this will iterate from the larger value to the
                        // smaller value.
                        if start_idx > stop_idx {
                            span_help_and_lint(cx, REVERSE_RANGE_LOOP, expr.span,
                                "this range is empty so this for loop will never run",
                                &format!("Consider using `({}..{}).rev()` if you are attempting to \
                                iterate over this range in reverse", stop_idx, start_idx));
                        } else if start_idx == stop_idx {
                            // if they are equal, it's also problematic - this loop
                            // will never run.
                            span_lint(cx, REVERSE_RANGE_LOOP, expr.span,
                                "this range is empty so this for loop will never run");
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
                    else if method_name == "next" &&
                            match_trait_method(cx, arg, &["core", "iter", "Iterator"]) {
                        span_lint(cx, ITER_NEXT_LOOP, expr.span,
                                  "you are iterating over `Iterator::next()` which is an Option; \
                                   this will compile but is probably not what you want");
                    }
                }
            }

            // Look for variables that are incremented once per loop iteration.
            let mut visitor = IncrementVisitor { cx: cx, states: HashMap::new(), depth: 0, done: false };
            walk_expr(&mut visitor, body);

            // For each candidate, check the parent block to see if
            // it's initialized to zero at the start of the loop.
            let map = &cx.tcx.map;
            let parent_scope = map.get_enclosing_scope(expr.id).and_then(|id| map.get_enclosing_scope(id) );
            if let Some(parent_id) = parent_scope {
                if let NodeBlock(block) = map.get(parent_id) {
                    for (id, _) in visitor.states.iter().filter( |&(_,v)| *v == VarState::IncrOnce) {
                        let mut visitor2 = InitializeVisitor { cx: cx, end_expr: expr, var_id: id.clone(),
                                                               state: VarState::IncrOnce, name: None,
                                                               depth: 0, done: false };
                        walk_block(&mut visitor2, block);

                        if visitor2.state == VarState::Warn {
                            if let Some(name) = visitor2.name {
                                span_lint(cx, EXPLICIT_COUNTER_LOOP, expr.span,
                                          &format!("the variable `{0}` is used as a loop counter. Consider \
                                                    using `for ({0}, item) in {1}.enumerate()` \
                                                    or similar iterators",
                                                   name, snippet(cx, arg.span, "_")));
                            }
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

    fn check_stmt(&mut self, cx: &Context, stmt: &Stmt) {
        if let StmtSemi(ref expr, _) = stmt.node {
            if let ExprMethodCall(ref method, _, ref args) = expr.node {
                if args.len() == 1 && method.node.name == "collect" &&
                        match_trait_method(cx, expr, &["core", "iter", "Iterator"]) {
                    span_lint(cx, UNUSED_COLLECT, expr.span, &format!(
                        "you are collect()ing an iterator and throwing away the result. \
                         Consider using an explicit for loop to exhaust the iterator"));
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
    // no walk_ptrs_ty: calling iter() on a reference can make sense because it
    // will allow further borrows afterwards
    let ty = cx.tcx.expr_ty(e);
    is_iterable_array(ty) ||
        match_type(cx, ty, &VEC_PATH) ||
        match_type(cx, ty, &LL_PATH) ||
        match_type(cx, ty, &["std", "collections", "hash", "map", "HashMap"]) ||
        match_type(cx, ty, &["std", "collections", "hash", "set", "HashSet"]) ||
        match_type(cx, ty, &["collections", "vec_deque", "VecDeque"]) ||
        match_type(cx, ty, &["collections", "binary_heap", "BinaryHeap"]) ||
        match_type(cx, ty, &["collections", "btree", "map", "BTreeMap"]) ||
        match_type(cx, ty, &["collections", "btree", "set", "BTreeSet"])
}

fn is_iterable_array(ty: ty::Ty) -> bool {
    //IntoIterator is currently only implemented for array sizes <= 32 in rustc
    match ty.sty {
        ty::TyArray(_, 0...32) => true,
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

// To trigger the EXPLICIT_COUNTER_LOOP lint, a variable must be
// incremented exactly once in the loop body, and initialized to zero
// at the start of the loop.
#[derive(PartialEq)]
enum VarState {
    Initial,      // Not examined yet
    IncrOnce,     // Incremented exactly once, may be a loop counter
    Declared,     // Declared but not (yet) initialized to zero
    Warn,
    DontWarn
}

// Scan a for loop for variables that are incremented exactly once.
struct IncrementVisitor<'v, 't: 'v> {
    cx: &'v Context<'v, 't>,      // context reference
    states: HashMap<NodeId, VarState>,  // incremented variables
    depth: u32,                         // depth of conditional expressions
    done: bool
}

impl<'v, 't> Visitor<'v> for IncrementVisitor<'v, 't> {
    fn visit_expr(&mut self, expr: &'v Expr) {
        if self.done {
            return;
        }

        // If node is a variable
        if let Some(def_id) = var_def_id(self.cx, expr) {
            if let Some(parent) = get_parent_expr(self.cx, expr) {
                let state = self.states.entry(def_id).or_insert(VarState::Initial);

                match parent.node {
                    ExprAssignOp(op, ref lhs, ref rhs) =>
                        if lhs.id == expr.id {
                            if op.node == BiAdd && is_integer_literal(rhs, 1) {
                                *state = match *state {
                                    VarState::Initial if self.depth == 0 => VarState::IncrOnce,
                                    _ => VarState::DontWarn
                                };
                            }
                            else {
                                // Assigned some other value
                                *state = VarState::DontWarn;
                            }
                        },
                    ExprAssign(ref lhs, _) if lhs.id == expr.id => *state = VarState::DontWarn,
                    ExprAddrOf(mutability,_) if mutability == MutMutable => *state = VarState::DontWarn,
                    _ => ()
                }
            }
        }
        // Give up if there are nested loops
        else if is_loop(expr) {
            self.states.clear();
            self.done = true;
            return;
        }
        // Keep track of whether we're inside a conditional expression
        else if is_conditional(expr) {
            self.depth += 1;
            walk_expr(self, expr);
            self.depth -= 1;
            return;
        }
        walk_expr(self, expr);
    }
}

// Check whether a variable is initialized to zero at the start of a loop.
struct InitializeVisitor<'v, 't: 'v> {
    cx: &'v Context<'v, 't>, // context reference
    end_expr: &'v Expr,      // the for loop. Stop scanning here.
    var_id: NodeId,
    state: VarState,
    name: Option<Name>,
    depth: u32,              // depth of conditional expressions
    done: bool
}

impl<'v, 't> Visitor<'v> for InitializeVisitor<'v, 't> {
    fn visit_decl(&mut self, decl: &'v Decl) {
        // Look for declarations of the variable
        if let DeclLocal(ref local) = decl.node {
            if local.pat.id == self.var_id {
                if let PatIdent(_, ref ident, _) = local.pat.node {
                    self.name = Some(ident.node.name);

                    self.state = if let Some(ref init) = local.init {
                        if is_integer_literal(init, 0) {
                            VarState::Warn
                        } else {
                            VarState::Declared
                        }
                    }
                    else {
                        VarState::Declared
                    }
                }
            }
        }
        walk_decl(self, decl);
    }

    fn visit_expr(&mut self, expr: &'v Expr) {
        if self.state == VarState::DontWarn || expr == self.end_expr {
            self.done = true;
        }
        // No need to visit expressions before the variable is
        // declared or after we've rejected it.
        if self.state == VarState::IncrOnce || self.done {
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
                        }},
                    ExprAddrOf(mutability,_) if mutability == MutMutable => self.state = VarState::DontWarn,
                    _ => ()
                }
            }
        }
        // If there are other loops between the declaration and the target loop, give up
        else if is_loop(expr) {
            self.state = VarState::DontWarn;
            self.done = true;
            return;
        }
        // Keep track of whether we're inside a conditional expression
        else if is_conditional(expr) {
            self.depth += 1;
            walk_expr(self, expr);
            self.depth -= 1;
            return;
        }
        walk_expr(self, expr);
    }
}

fn var_def_id(cx: &Context, expr: &Expr) -> Option<NodeId> {
    if let Some(path_res) = cx.tcx.def_map.borrow().get(&expr.id) {
        if let DefLocal(node_id) = path_res.base_def {
            return Some(node_id)
        }
    }
    None
}

fn is_loop(expr: &Expr) -> bool {
    match expr.node {
        ExprLoop(..) | ExprWhile(..)  => true,
        _ => false
    }
}

fn is_conditional(expr: &Expr) -> bool {
    match expr.node {
        ExprIf(..) | ExprMatch(..) => true,
        _ => false
    }
}

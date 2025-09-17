//! This file builds up the `ScopeTree`, which describes
//! the parent links in the region hierarchy.
//!
//! For more information about how MIR-based region-checking works,
//! see the [rustc dev guide].
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/borrow_check.html

use std::mem;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Arm, Block, Expr, LetStmt, Pat, PatKind, Stmt};
use rustc_index::Idx;
use rustc_middle::middle::region::*;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint;
use rustc_span::source_map;
use tracing::debug;

#[derive(Debug, Copy, Clone)]
struct Context {
    /// The scope that contains any new variables declared.
    var_parent: Option<Scope>,

    /// Region parent of expressions, etc.
    parent: Option<Scope>,
}

struct ScopeResolutionVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    // The generated scope tree.
    scope_tree: ScopeTree,

    cx: Context,

    extended_super_lets: FxHashMap<hir::ItemLocalId, Option<Scope>>,
}

/// Records the lifetime of a local variable as `cx.var_parent`
fn record_var_lifetime(visitor: &mut ScopeResolutionVisitor<'_>, var_id: hir::ItemLocalId) {
    match visitor.cx.var_parent {
        None => {
            // this can happen in extern fn declarations like
            //
            // extern fn isalnum(c: c_int) -> c_int
        }
        Some(parent_scope) => visitor.scope_tree.record_var_scope(var_id, parent_scope),
    }
}

fn resolve_block<'tcx>(
    visitor: &mut ScopeResolutionVisitor<'tcx>,
    blk: &'tcx hir::Block<'tcx>,
    terminating: bool,
) {
    debug!("resolve_block(blk.hir_id={:?})", blk.hir_id);

    let prev_cx = visitor.cx;

    // We treat the tail expression in the block (if any) somewhat
    // differently from the statements. The issue has to do with
    // temporary lifetimes. Consider the following:
    //
    //    quux({
    //        let inner = ... (&bar()) ...;
    //
    //        (... (&foo()) ...) // (the tail expression)
    //    }, other_argument());
    //
    // Each of the statements within the block is a terminating
    // scope, and thus a temporary (e.g., the result of calling
    // `bar()` in the initializer expression for `let inner = ...;`)
    // will be cleaned up immediately after its corresponding
    // statement (i.e., `let inner = ...;`) executes.
    //
    // On the other hand, temporaries associated with evaluating the
    // tail expression for the block are assigned lifetimes so that
    // they will be cleaned up as part of the terminating scope
    // *surrounding* the block expression. Here, the terminating
    // scope for the block expression is the `quux(..)` call; so
    // those temporaries will only be cleaned up *after* both
    // `other_argument()` has run and also the call to `quux(..)`
    // itself has returned.

    visitor.enter_node_scope_with_dtor(blk.hir_id.local_id, terminating);
    visitor.cx.var_parent = visitor.cx.parent;

    {
        // This block should be kept approximately in sync with
        // `intravisit::walk_block`. (We manually walk the block, rather
        // than call `walk_block`, in order to maintain precise
        // index information.)

        for (i, statement) in blk.stmts.iter().enumerate() {
            match statement.kind {
                hir::StmtKind::Let(LetStmt { els: Some(els), .. }) => {
                    // Let-else has a special lexical structure for variables.
                    // First we take a checkpoint of the current scope context here.
                    let mut prev_cx = visitor.cx;

                    visitor.enter_scope(Scope {
                        local_id: blk.hir_id.local_id,
                        data: ScopeData::Remainder(FirstStatementIndex::new(i)),
                    });
                    visitor.cx.var_parent = visitor.cx.parent;
                    visitor.visit_stmt(statement);
                    // We need to back out temporarily to the last enclosing scope
                    // for the `else` block, so that even the temporaries receiving
                    // extended lifetime will be dropped inside this block.
                    // We are visiting the `else` block in this order so that
                    // the sequence of visits agree with the order in the default
                    // `hir::intravisit` visitor.
                    mem::swap(&mut prev_cx, &mut visitor.cx);
                    resolve_block(visitor, els, true);
                    // From now on, we continue normally.
                    visitor.cx = prev_cx;
                }
                hir::StmtKind::Let(..) => {
                    // Each declaration introduces a subscope for bindings
                    // introduced by the declaration; this subscope covers a
                    // suffix of the block. Each subscope in a block has the
                    // previous subscope in the block as a parent, except for
                    // the first such subscope, which has the block itself as a
                    // parent.
                    visitor.enter_scope(Scope {
                        local_id: blk.hir_id.local_id,
                        data: ScopeData::Remainder(FirstStatementIndex::new(i)),
                    });
                    visitor.cx.var_parent = visitor.cx.parent;
                    visitor.visit_stmt(statement)
                }
                hir::StmtKind::Item(..) => {
                    // Don't create scopes for items, since they won't be
                    // lowered to THIR and MIR.
                }
                hir::StmtKind::Expr(..) | hir::StmtKind::Semi(..) => visitor.visit_stmt(statement),
            }
        }
        if let Some(tail_expr) = blk.expr {
            let local_id = tail_expr.hir_id.local_id;
            let edition = blk.span.edition();
            let terminating = edition.at_least_rust_2024();
            if !terminating
                && !visitor
                    .tcx
                    .lints_that_dont_need_to_run(())
                    .contains(&lint::LintId::of(lint::builtin::TAIL_EXPR_DROP_ORDER))
            {
                // If this temporary scope will be changing once the codebase adopts Rust 2024,
                // and we are linting about possible semantic changes that would result,
                // then record this node-id in the field `backwards_incompatible_scope`
                // for future reference.
                visitor
                    .scope_tree
                    .backwards_incompatible_scope
                    .insert(local_id, Scope { local_id, data: ScopeData::Node });
            }
            resolve_expr(visitor, tail_expr, terminating);
        }
    }

    visitor.cx = prev_cx;
}

/// Resolve a condition from an `if` expression or match guard so that it is a terminating scope
/// if it doesn't contain `let` expressions.
fn resolve_cond<'tcx>(visitor: &mut ScopeResolutionVisitor<'tcx>, cond: &'tcx hir::Expr<'tcx>) {
    let terminate = match cond.kind {
        // Temporaries for `let` expressions must live into the success branch.
        hir::ExprKind::Let(_) => false,
        // Logical operator chains are handled in `resolve_expr`. Since logical operator chains in
        // conditions are lowered to control-flow rather than boolean temporaries, there's no
        // temporary to drop for logical operators themselves. `resolve_expr` will also recursively
        // wrap any operands in terminating scopes, other than `let` expressions (which we shouldn't
        // terminate) and other logical operators (which don't need a terminating scope, since their
        // operands will be terminated). Any temporaries that would need to be dropped will be
        // dropped before we leave this operator's scope; terminating them here would be redundant.
        hir::ExprKind::Binary(
            source_map::Spanned { node: hir::BinOpKind::And | hir::BinOpKind::Or, .. },
            _,
            _,
        ) => false,
        // Otherwise, conditions should always drop their temporaries.
        _ => true,
    };
    resolve_expr(visitor, cond, terminate);
}

fn resolve_arm<'tcx>(visitor: &mut ScopeResolutionVisitor<'tcx>, arm: &'tcx hir::Arm<'tcx>) {
    let prev_cx = visitor.cx;

    visitor.enter_node_scope_with_dtor(arm.hir_id.local_id, true);
    visitor.cx.var_parent = visitor.cx.parent;

    resolve_pat(visitor, arm.pat);
    if let Some(guard) = arm.guard {
        // We introduce a new scope to contain bindings and temporaries from `if let` guards, to
        // ensure they're dropped before the arm's pattern's bindings. This extends to the end of
        // the arm body and is the scope of its locals as well.
        visitor.enter_scope(Scope { local_id: arm.hir_id.local_id, data: ScopeData::MatchGuard });
        visitor.cx.var_parent = visitor.cx.parent;
        resolve_cond(visitor, guard);
    }
    resolve_expr(visitor, arm.body, false);

    visitor.cx = prev_cx;
}

#[tracing::instrument(level = "debug", skip(visitor))]
fn resolve_pat<'tcx>(visitor: &mut ScopeResolutionVisitor<'tcx>, pat: &'tcx hir::Pat<'tcx>) {
    // If this is a binding then record the lifetime of that binding.
    if let PatKind::Binding(..) = pat.kind {
        record_var_lifetime(visitor, pat.hir_id.local_id);
    }

    intravisit::walk_pat(visitor, pat);
}

fn resolve_stmt<'tcx>(visitor: &mut ScopeResolutionVisitor<'tcx>, stmt: &'tcx hir::Stmt<'tcx>) {
    let stmt_id = stmt.hir_id.local_id;
    debug!("resolve_stmt(stmt.id={:?})", stmt_id);

    if let hir::StmtKind::Let(LetStmt { super_: Some(_), .. }) = stmt.kind {
        // `super let` statement does not start a new scope, such that
        //
        //     { super let x = identity(&temp()); &x }.method();
        //
        // behaves exactly as
        //
        //     (&identity(&temp()).method();
        intravisit::walk_stmt(visitor, stmt);
    } else {
        // Every statement will clean up the temporaries created during
        // execution of that statement. Therefore each statement has an
        // associated destruction scope that represents the scope of the
        // statement plus its destructors, and thus the scope for which
        // regions referenced by the destructors need to survive.

        let prev_parent = visitor.cx.parent;
        visitor.enter_node_scope_with_dtor(stmt_id, true);

        intravisit::walk_stmt(visitor, stmt);

        visitor.cx.parent = prev_parent;
    }
}

#[tracing::instrument(level = "debug", skip(visitor))]
fn resolve_expr<'tcx>(
    visitor: &mut ScopeResolutionVisitor<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    terminating: bool,
) {
    let prev_cx = visitor.cx;
    visitor.enter_node_scope_with_dtor(expr.hir_id.local_id, terminating);

    match expr.kind {
        // Conditional or repeating scopes are always terminating
        // scopes, meaning that temporaries cannot outlive them.
        // This ensures fixed size stacks.
        hir::ExprKind::Binary(
            source_map::Spanned { node: hir::BinOpKind::And | hir::BinOpKind::Or, .. },
            left,
            right,
        ) => {
            // expr is a short circuiting operator (|| or &&). As its
            // functionality can't be overridden by traits, it always
            // processes bool sub-expressions. bools are Copy and thus we
            // can drop any temporaries in evaluation (read) order
            // (with the exception of potentially failing let expressions).
            // We achieve this by enclosing the operands in a terminating
            // scope, both the LHS and the RHS.

            // We optimize this a little in the presence of chains.
            // Chains like a && b && c get lowered to AND(AND(a, b), c).
            // In here, b and c are RHS, while a is the only LHS operand in
            // that chain. This holds true for longer chains as well: the
            // leading operand is always the only LHS operand that is not a
            // binop itself. Putting a binop like AND(a, b) into a
            // terminating scope is not useful, thus we only put the LHS
            // into a terminating scope if it is not a binop.

            let terminate_lhs = match left.kind {
                // let expressions can create temporaries that live on
                hir::ExprKind::Let(_) => false,
                // binops already drop their temporaries, so there is no
                // need to put them into a terminating scope.
                // This is purely an optimization to reduce the number of
                // terminating scopes.
                hir::ExprKind::Binary(
                    source_map::Spanned { node: hir::BinOpKind::And | hir::BinOpKind::Or, .. },
                    ..,
                ) => false,
                // otherwise: mark it as terminating
                _ => true,
            };

            // `Let` expressions (in a let-chain) shouldn't be terminating, as their temporaries
            // should live beyond the immediate expression
            let terminate_rhs = !matches!(right.kind, hir::ExprKind::Let(_));

            resolve_expr(visitor, left, terminate_lhs);
            resolve_expr(visitor, right, terminate_rhs);
        }
        // Manually recurse over closures, because they are nested bodies
        // that share the parent environment. We handle const blocks in
        // `visit_inline_const`.
        hir::ExprKind::Closure(&hir::Closure { body, .. }) => {
            let body = visitor.tcx.hir_body(body);
            visitor.visit_body(body);
        }
        // Ordinarily, we can rely on the visit order of HIR intravisit
        // to correspond to the actual execution order of statements.
        // However, there's a weird corner case with compound assignment
        // operators (e.g. `a += b`). The evaluation order depends on whether
        // or not the operator is overloaded (e.g. whether or not a trait
        // like AddAssign is implemented).
        //
        // For primitive types (which, despite having a trait impl, don't actually
        // end up calling it), the evaluation order is right-to-left. For example,
        // the following code snippet:
        //
        //    let y = &mut 0;
        //    *{println!("LHS!"); y} += {println!("RHS!"); 1};
        //
        // will print:
        //
        // RHS!
        // LHS!
        //
        // However, if the operator is used on a non-primitive type,
        // the evaluation order will be left-to-right, since the operator
        // actually get desugared to a method call. For example, this
        // nearly identical code snippet:
        //
        //     let y = &mut String::new();
        //    *{println!("LHS String"); y} += {println!("RHS String"); "hi"};
        //
        // will print:
        // LHS String
        // RHS String
        //
        // To determine the actual execution order, we need to perform
        // trait resolution. Fortunately, we don't need to know the actual execution order.
        hir::ExprKind::AssignOp(_, left_expr, right_expr) => {
            visitor.visit_expr(right_expr);
            visitor.visit_expr(left_expr);
        }

        hir::ExprKind::If(cond, then, Some(otherwise)) => {
            let expr_cx = visitor.cx;
            let data = if expr.span.at_least_rust_2024() {
                ScopeData::IfThenRescope
            } else {
                ScopeData::IfThen
            };
            visitor.enter_scope(Scope { local_id: then.hir_id.local_id, data });
            visitor.cx.var_parent = visitor.cx.parent;
            resolve_cond(visitor, cond);
            resolve_expr(visitor, then, true);
            visitor.cx = expr_cx;
            resolve_expr(visitor, otherwise, true);
        }

        hir::ExprKind::If(cond, then, None) => {
            let expr_cx = visitor.cx;
            let data = if expr.span.at_least_rust_2024() {
                ScopeData::IfThenRescope
            } else {
                ScopeData::IfThen
            };
            visitor.enter_scope(Scope { local_id: then.hir_id.local_id, data });
            visitor.cx.var_parent = visitor.cx.parent;
            resolve_cond(visitor, cond);
            resolve_expr(visitor, then, true);
            visitor.cx = expr_cx;
        }

        hir::ExprKind::Loop(body, _, _, _) => {
            resolve_block(visitor, body, true);
        }

        hir::ExprKind::DropTemps(expr) => {
            // `DropTemps(expr)` does not denote a conditional scope.
            // Rather, we want to achieve the same behavior as `{ let _t = expr; _t }`.
            resolve_expr(visitor, expr, true);
        }

        _ => intravisit::walk_expr(visitor, expr),
    }

    visitor.cx = prev_cx;
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum LetKind {
    Regular,
    Super,
}

fn resolve_local<'tcx>(
    visitor: &mut ScopeResolutionVisitor<'tcx>,
    pat: Option<&'tcx hir::Pat<'tcx>>,
    init: Option<&'tcx hir::Expr<'tcx>>,
    let_kind: LetKind,
) {
    debug!("resolve_local(pat={:?}, init={:?}, let_kind={:?})", pat, init, let_kind);

    // As an exception to the normal rules governing temporary
    // lifetimes, initializers in a let have a temporary lifetime
    // of the enclosing block. This means that e.g., a program
    // like the following is legal:
    //
    //     let ref x = HashMap::new();
    //
    // Because the hash map will be freed in the enclosing block.
    //
    // We express the rules more formally based on 3 grammars (defined
    // fully in the helpers below that implement them):
    //
    // 1. `E&`, which matches expressions like `&<rvalue>` that
    //    own a pointer into the stack.
    //
    // 2. `P&`, which matches patterns like `ref x` or `(ref x, ref
    //    y)` that produce ref bindings into the value they are
    //    matched against or something (at least partially) owned by
    //    the value they are matched against. (By partially owned,
    //    I mean that creating a binding into a ref-counted or managed value
    //    would still count.)
    //
    // 3. `ET`, which matches both rvalues like `foo()` as well as places
    //    based on rvalues like `foo().x[2].y`.
    //
    // A subexpression `<rvalue>` that appears in a let initializer
    // `let pat [: ty] = expr` has an extended temporary lifetime if
    // any of the following conditions are met:
    //
    // A. `pat` matches `P&` and `expr` matches `ET`
    //    (covers cases where `pat` creates ref bindings into an rvalue
    //     produced by `expr`)
    // B. `ty` is a borrowed pointer and `expr` matches `ET`
    //    (covers cases where coercion creates a borrow)
    // C. `expr` matches `E&`
    //    (covers cases `expr` borrows an rvalue that is then assigned
    //     to memory (at least partially) owned by the binding)
    //
    // Here are some examples hopefully giving an intuition where each
    // rule comes into play and why:
    //
    // Rule A. `let (ref x, ref y) = (foo().x, 44)`. The rvalue `(22, 44)`
    // would have an extended lifetime, but not `foo()`.
    //
    // Rule B. `let x = &foo().x`. The rvalue `foo()` would have extended
    // lifetime.
    //
    // In some cases, multiple rules may apply (though not to the same
    // rvalue). For example:
    //
    //     let ref x = [&a(), &b()];
    //
    // Here, the expression `[...]` has an extended lifetime due to rule
    // A, but the inner rvalues `a()` and `b()` have an extended lifetime
    // due to rule C.

    let extend_initializer = match let_kind {
        LetKind::Regular => true,
        LetKind::Super
            if let Some(scope) =
                visitor.extended_super_lets.remove(&pat.unwrap().hir_id.local_id) =>
        {
            // This expression was lifetime-extended by a parent let binding. E.g.
            //
            //     let a = {
            //         super let b = temp();
            //         &b
            //     };
            //
            // (Which needs to behave exactly as: let a = &temp();)
            //
            // Processing of `let a` will have already decided to extend the lifetime of this
            // `super let` to its own var_scope. We use that scope.
            visitor.cx.var_parent = scope;
            // Extend temporaries to live in the same scope as the parent `let`'s bindings.
            true
        }
        LetKind::Super => {
            // This `super let` is not subject to lifetime extension from a parent let binding. E.g.
            //
            //     identity({ super let x = temp(); &x }).method();
            //
            // (Which needs to behave exactly as: identity(&temp()).method();)
            //
            // Iterate up to the enclosing destruction scope to find the same scope that will also
            // be used for the result of the block itself.
            if let Some(inner_scope) = visitor.cx.var_parent {
                (visitor.cx.var_parent, _) = visitor.scope_tree.default_temporary_scope(inner_scope)
            }
            // Don't lifetime-extend child `super let`s or block tail expressions' temporaries in
            // the initializer when this `super let` is not itself extended by a parent `let`
            // (#145784). Block tail expressions are temporary drop scopes in Editions 2024 and
            // later, their temps shouldn't outlive the block in e.g. `f(pin!({ &temp() }))`.
            false
        }
    };

    if let Some(expr) = init
        && extend_initializer
    {
        record_rvalue_scope_if_borrow_expr(visitor, expr, visitor.cx.var_parent);

        if let Some(pat) = pat {
            if is_binding_pat(pat) {
                visitor.scope_tree.record_rvalue_candidate(
                    expr.hir_id,
                    RvalueCandidate {
                        target: expr.hir_id.local_id,
                        lifetime: visitor.cx.var_parent,
                    },
                );
            }
        }
    }

    // Make sure we visit the initializer first.
    // The correct order, as shared between drop_ranges and intravisitor,
    // is to walk initializer, followed by pattern bindings, finally followed by the `else` block.
    if let Some(expr) = init {
        visitor.visit_expr(expr);
    }

    if let Some(pat) = pat {
        visitor.visit_pat(pat);
    }

    /// Returns `true` if `pat` match the `P&` non-terminal.
    ///
    /// ```text
    ///     P& = ref X
    ///        | StructName { ..., P&, ... }
    ///        | VariantName(..., P&, ...)
    ///        | [ ..., P&, ... ]
    ///        | ( ..., P&, ... )
    ///        | ... "|" P& "|" ...
    ///        | box P&
    ///        | P& if ...
    /// ```
    fn is_binding_pat(pat: &hir::Pat<'_>) -> bool {
        // Note that the code below looks for *explicit* refs only, that is, it won't
        // know about *implicit* refs as introduced in #42640.
        //
        // This is not a problem. For example, consider
        //
        //      let (ref x, ref y) = (Foo { .. }, Bar { .. });
        //
        // Due to the explicit refs on the left hand side, the below code would signal
        // that the temporary value on the right hand side should live until the end of
        // the enclosing block (as opposed to being dropped after the let is complete).
        //
        // To create an implicit ref, however, you must have a borrowed value on the RHS
        // already, as in this example (which won't compile before #42640):
        //
        //      let Foo { x, .. } = &Foo { x: ..., ... };
        //
        // in place of
        //
        //      let Foo { ref x, .. } = Foo { ... };
        //
        // In the former case (the implicit ref version), the temporary is created by the
        // & expression, and its lifetime would be extended to the end of the block (due
        // to a different rule, not the below code).
        match pat.kind {
            PatKind::Binding(hir::BindingMode(hir::ByRef::Yes(_), _), ..) => true,

            PatKind::Struct(_, field_pats, _) => field_pats.iter().any(|fp| is_binding_pat(fp.pat)),

            PatKind::Slice(pats1, pats2, pats3) => {
                pats1.iter().any(|p| is_binding_pat(p))
                    || pats2.iter().any(|p| is_binding_pat(p))
                    || pats3.iter().any(|p| is_binding_pat(p))
            }

            PatKind::Or(subpats)
            | PatKind::TupleStruct(_, subpats, _)
            | PatKind::Tuple(subpats, _) => subpats.iter().any(|p| is_binding_pat(p)),

            PatKind::Box(subpat) | PatKind::Deref(subpat) | PatKind::Guard(subpat, _) => {
                is_binding_pat(subpat)
            }

            PatKind::Ref(_, _)
            | PatKind::Binding(hir::BindingMode(hir::ByRef::No, _), ..)
            | PatKind::Missing
            | PatKind::Wild
            | PatKind::Never
            | PatKind::Expr(_)
            | PatKind::Range(_, _, _)
            | PatKind::Err(_) => false,
        }
    }

    /// If `expr` matches the `E&` grammar, then records an extended rvalue scope as appropriate:
    ///
    /// ```text
    ///     E& = & ET
    ///        | StructName { ..., f: E&, ... }
    ///        | [ ..., E&, ... ]
    ///        | ( ..., E&, ... )
    ///        | {...; E&}
    ///        | { super let ... = E&; ... }
    ///        | if _ { ...; E& } else { ...; E& }
    ///        | match _ { ..., _ => E&, ... }
    ///        | box E&
    ///        | E& as ...
    ///        | ( E& )
    /// ```
    fn record_rvalue_scope_if_borrow_expr<'tcx>(
        visitor: &mut ScopeResolutionVisitor<'tcx>,
        expr: &hir::Expr<'_>,
        blk_id: Option<Scope>,
    ) {
        match expr.kind {
            hir::ExprKind::AddrOf(_, _, subexpr) => {
                record_rvalue_scope_if_borrow_expr(visitor, subexpr, blk_id);
                visitor.scope_tree.record_rvalue_candidate(
                    subexpr.hir_id,
                    RvalueCandidate { target: subexpr.hir_id.local_id, lifetime: blk_id },
                );
            }
            hir::ExprKind::Struct(_, fields, _) => {
                for field in fields {
                    record_rvalue_scope_if_borrow_expr(visitor, field.expr, blk_id);
                }
            }
            hir::ExprKind::Array(subexprs) | hir::ExprKind::Tup(subexprs) => {
                for subexpr in subexprs {
                    record_rvalue_scope_if_borrow_expr(visitor, subexpr, blk_id);
                }
            }
            hir::ExprKind::Cast(subexpr, _) => {
                record_rvalue_scope_if_borrow_expr(visitor, subexpr, blk_id)
            }
            hir::ExprKind::Block(block, _) => {
                if let Some(subexpr) = block.expr {
                    record_rvalue_scope_if_borrow_expr(visitor, subexpr, blk_id);
                }
                for stmt in block.stmts {
                    if let hir::StmtKind::Let(local) = stmt.kind
                        && let Some(_) = local.super_
                    {
                        visitor.extended_super_lets.insert(local.pat.hir_id.local_id, blk_id);
                    }
                }
            }
            hir::ExprKind::If(_, then_block, else_block) => {
                record_rvalue_scope_if_borrow_expr(visitor, then_block, blk_id);
                if let Some(else_block) = else_block {
                    record_rvalue_scope_if_borrow_expr(visitor, else_block, blk_id);
                }
            }
            hir::ExprKind::Match(_, arms, _) => {
                for arm in arms {
                    record_rvalue_scope_if_borrow_expr(visitor, arm.body, blk_id);
                }
            }
            hir::ExprKind::Call(func, args) => {
                // Recurse into tuple constructors, such as `Some(&temp())`.
                //
                // That way, there is no difference between `Some(..)` and `Some { 0: .. }`,
                // even though the former is syntactically a function call.
                if let hir::ExprKind::Path(path) = &func.kind
                    && let hir::QPath::Resolved(None, path) = path
                    && let Res::SelfCtor(_) | Res::Def(DefKind::Ctor(_, CtorKind::Fn), _) = path.res
                {
                    for arg in args {
                        record_rvalue_scope_if_borrow_expr(visitor, arg, blk_id);
                    }
                }
            }
            _ => {}
        }
    }
}

impl<'tcx> ScopeResolutionVisitor<'tcx> {
    /// Records the current parent (if any) as the parent of `child_scope`.
    fn record_child_scope(&mut self, child_scope: Scope) {
        let parent = self.cx.parent;
        self.scope_tree.record_scope_parent(child_scope, parent);
    }

    /// Records the current parent (if any) as the parent of `child_scope`,
    /// and sets `child_scope` as the new current parent.
    fn enter_scope(&mut self, child_scope: Scope) {
        self.record_child_scope(child_scope);
        self.cx.parent = Some(child_scope);
    }

    fn enter_node_scope_with_dtor(&mut self, id: hir::ItemLocalId, terminating: bool) {
        // If node was previously marked as a terminating scope during the
        // recursive visit of its parent node in the HIR, then we need to
        // account for the destruction scope representing the scope of
        // the destructors that run immediately after it completes.
        if terminating {
            self.enter_scope(Scope { local_id: id, data: ScopeData::Destruction });
        }
        self.enter_scope(Scope { local_id: id, data: ScopeData::Node });
    }

    fn enter_body(&mut self, hir_id: hir::HirId, f: impl FnOnce(&mut Self)) {
        let outer_cx = self.cx;

        self.enter_scope(Scope { local_id: hir_id.local_id, data: ScopeData::CallSite });
        self.enter_scope(Scope { local_id: hir_id.local_id, data: ScopeData::Arguments });

        f(self);

        // Restore context we had at the start.
        self.cx = outer_cx;
    }
}

impl<'tcx> Visitor<'tcx> for ScopeResolutionVisitor<'tcx> {
    fn visit_block(&mut self, b: &'tcx Block<'tcx>) {
        resolve_block(self, b, false);
    }

    fn visit_body(&mut self, body: &hir::Body<'tcx>) {
        let body_id = body.id();
        let owner_id = self.tcx.hir_body_owner_def_id(body_id);

        debug!(
            "visit_body(id={:?}, span={:?}, body.id={:?}, cx.parent={:?})",
            owner_id,
            self.tcx.sess.source_map().span_to_diagnostic_string(body.value.span),
            body_id,
            self.cx.parent
        );

        self.enter_body(body.value.hir_id, |this| {
            if this.tcx.hir_body_owner_kind(owner_id).is_fn_or_closure() {
                // The arguments and `self` are parented to the fn.
                this.cx.var_parent = this.cx.parent;
                for param in body.params {
                    this.visit_pat(param.pat);
                }

                // The body of the every fn is a root scope.
                resolve_expr(this, body.value, true);
            } else {
                // Only functions have an outer terminating (drop) scope, while
                // temporaries in constant initializers may be 'static, but only
                // according to rvalue lifetime semantics, using the same
                // syntactical rules used for let initializers.
                //
                // e.g., in `let x = &f();`, the temporary holding the result from
                // the `f()` call lives for the entirety of the surrounding block.
                //
                // Similarly, `const X: ... = &f();` would have the result of `f()`
                // live for `'static`, implying (if Drop restrictions on constants
                // ever get lifted) that the value *could* have a destructor, but
                // it'd get leaked instead of the destructor running during the
                // evaluation of `X` (if at all allowed by CTFE).
                //
                // However, `const Y: ... = g(&f());`, like `let y = g(&f());`,
                // would *not* let the `f()` temporary escape into an outer scope
                // (i.e., `'static`), which means that after `g` returns, it drops,
                // and all the associated destruction scope rules apply.
                this.cx.var_parent = None;
                this.enter_scope(Scope {
                    local_id: body.value.hir_id.local_id,
                    data: ScopeData::Destruction,
                });
                resolve_local(this, None, Some(body.value), LetKind::Regular);
            }
        })
    }

    fn visit_arm(&mut self, a: &'tcx Arm<'tcx>) {
        resolve_arm(self, a);
    }
    fn visit_pat(&mut self, p: &'tcx Pat<'tcx>) {
        resolve_pat(self, p);
    }
    fn visit_stmt(&mut self, s: &'tcx Stmt<'tcx>) {
        resolve_stmt(self, s);
    }
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        resolve_expr(self, ex, false);
    }
    fn visit_local(&mut self, l: &'tcx LetStmt<'tcx>) {
        let let_kind = match l.super_ {
            Some(_) => LetKind::Super,
            None => LetKind::Regular,
        };
        resolve_local(self, Some(l.pat), l.init, let_kind);
    }
    fn visit_inline_const(&mut self, c: &'tcx hir::ConstBlock) {
        let body = self.tcx.hir_body(c.body);
        self.visit_body(body);
    }
}

/// Per-body `region::ScopeTree`. The `DefId` should be the owner `DefId` for the body;
/// in the case of closures, this will be redirected to the enclosing function.
///
/// Performance: This is a query rather than a simple function to enable
/// re-use in incremental scenarios. We may sometimes need to rerun the
/// type checker even when the HIR hasn't changed, and in those cases
/// we can avoid reconstructing the region scope tree.
pub(crate) fn region_scope_tree(tcx: TyCtxt<'_>, def_id: DefId) -> &ScopeTree {
    let typeck_root_def_id = tcx.typeck_root_def_id(def_id);
    if typeck_root_def_id != def_id {
        return tcx.region_scope_tree(typeck_root_def_id);
    }

    let scope_tree = if let Some(body) = tcx.hir_maybe_body_owned_by(def_id.expect_local()) {
        let mut visitor = ScopeResolutionVisitor {
            tcx,
            scope_tree: ScopeTree::default(),
            cx: Context { parent: None, var_parent: None },
            extended_super_lets: Default::default(),
        };

        visitor.scope_tree.root_body = Some(body.value.hir_id);
        visitor.visit_body(&body);
        visitor.scope_tree
    } else {
        ScopeTree::default()
    };

    tcx.arena.alloc(scope_tree)
}

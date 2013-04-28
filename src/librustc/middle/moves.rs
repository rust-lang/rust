// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Moves Computation

The goal of this file is to compute which
expressions/patterns/captures correspond to *moves*.  This is
generally a function of the context in which the expression appears as
well as the expression's type.

## Examples

We will use the following fragment of code to explain the various
considerations.  Note that in this code `x` is used after it has been
moved here.  This is not relevant to this pass, though the information
we compute would later be used to detect this error (see the section
Enforcement of Moves, below).

    struct Foo { a: int, b: ~int }
    let x: Foo = ...;
    let w = (x {Read}).a;      // Read
    let y = (x {Move}).b;      // Move
    let z = copy (x {Read}).b; // Read

Let's look at these examples one by one.  In the first case, `w`, the
expression being assigned is `x.a`, which has `int` type.  In that
case, the value is read, and the container (`x`) is also read.

In the second case, `y`, `x.b` is being assigned which has type
`~int`.  Because this type moves by default, that will be a move
reference.  Whenever we move from a compound expression like `x.b` (or
`x[b]` or `*x` or `{x)[b].c`, etc), this invalidates all containing
expressions since we do not currently permit "incomplete" variables
where part of them has been moved and part has not.  In this case,
this means that the reference to `x` is also a move.  We'll see later,
though, that these kind of "partial moves", where part of the
expression has been moved, are classified and stored somewhat
differently.

The final example (`z`) is `copy x.b`: in this case, although the
expression being assigned has type `~int`, there are no moves
involved.

### Patterns

For each binding in a match or let pattern, we also compute a read
or move designation.  A move binding means that the value will be
moved from the value being matched.  As a result, the expression
being matched (aka, the 'discriminant') is either moved or read
depending on whethe the bindings move the value they bind to out of
the discriminant.

For examples, consider this match expression:

    match x {Move} {
      Foo { a: a {Read}, b: b {Move} } => {...}
    }

Here, the binding `b` is value (not ref) mode, and `b` has type
`~int`, and therefore the discriminant expression `x` would be
incomplete so it also considered moved.

In the following two examples, in contrast, the mode of `b` is either
`copy` or `ref` and hence the overall result is a read:

    match x {Read} {
      Foo { a: a {Read}, b: copy b {Read} } => {...}
    }

    match x {Read} {
      Foo { a: a {Read}, b: ref b {Read} } => {...}
    }

Similar reasoning can be applied to `let` expressions:

    let Foo { a: a {Read}, b: b {Move} } = x {Move};
    let Foo { a: a {Read}, b: copy b {Read} } = x {Read};
    let Foo { a: a {Read}, b: ref b  {Read} } = x {Read};

## Output

The pass results in the struct `MoveMaps` which contains two sets,
`moves_map` and `variable_moves_map`, and one map, `capture_map`.

`moves_map` is a set containing the id of every *outermost
expression* or *binding* that is moved.  Note that `moves_map` only
contains the *outermost expressions* that are moved.  Therefore, if
you have a use of `x.b`, as in the example `y` above, the
expression `x.b` would be in the `moves_map` but not `x`.  The
reason for this is that, for most purposes, it's only the outermost
expression that is needed.  The borrow checker and trans, for
example, only care about the outermost expressions that are moved.
It is more efficient therefore just to store those entries.

In the case of the liveness pass, however, we need to know which
*variable references* are moved (see the Enforcement of Moves
section below for more details).  That is, for the `x.b`
expression, liveness only cares about the `x`.  For this purpose,
we have a second map, `variable_moves_map`, that contains the ids
of all variable references which is moved.

The `capture_map` maps from the node_id of a closure expression to an
array of `CaptureVar` structs detailing which variables are captured
and how (by ref, by copy, by move).

## Enforcement of Moves

The enforcement of moves is somewhat complicated because it is divided
amongst the liveness and borrowck modules. In general, the borrow
checker is responsible for guaranteeing that *only owned data is
moved*.  The liveness checker, in contrast, is responsible for
checking that *no variable is used after it is moved*.

To see the difference, let's look at a few examples.  Here is a
program fragment where the error would be caught by liveness:

    struct Foo { a: int, b: ~int }
    let x: Foo = ...;
    let y = x.b; // (1)
    let z = x;   // (2)            //~ ERROR use of moved value `x`

Here the liveness checker will see the assignment to `y` moves
invalidates the variable `x` because it moves the expression `x.b`.
An error is resported because `x` is not dead at the point where it is
invalidated.

In more concrete terms, the `moves_map` generated from this example
would contain both the expression `x.b` (1) and the expression `x`
(2).  Note that it would not contain `x` (1), because `moves_map` only
contains the outermost expressions that are moved.  However,
`moves_map` is not used by liveness.  It uses the
`variable_moves_map`, which would contain both references to `x`: (1)
and (2).  Therefore, after computing which variables are live where,
liveness will see that the reference (1) to `x` is both present in
`variable_moves_map` and that `x` is live and report an error.

Now let's look at another illegal example, but one where liveness would
not catch the error:

    struct Foo { a: int, b: ~int }
    let x: @Foo = ...;
    let y = x.b;                   //~ ERROR move from managed (@) box

This is an interesting example because the only change I've made is
to make `x` have type `@Foo` and not `Foo`.  Thanks to auto-deref,
the expression `x.b` still works, but now it is short for `{x).b`,
and hence the move is actually moving out of the contents of a
managed box, which is illegal.  However, liveness knows nothing of
this.  It only tracks what variables are used where.  The moves
pass (that is, this pass) is also ignorant of such details.  From
the perspective of the moves pass, the `let y = x.b` line above
will be categorized as follows:

    let y = {(x{Move}) {Move}).b; {Move}

Therefore, the reference to `x` will be present in
`variable_moves_map`, but liveness will not report an error because
there is no subsequent use.

This is where the borrow checker comes in.  When the borrow checker
runs, it will see that `x.b` is present in the `moves_map`.  It will
use the `mem_categorization` module to determine where the result of
this expression resides in memory and see that it is owned by managed
data, and report an error.

In principle, liveness could use the `mem_categorization` module
itself and check that moves always originate from owned data
(historically, of course, this was not the case; `mem_categorization`
used to be private to the borrow checker).  However, there is another
kind of error which liveness could not possibly detect. Sometimes a
move is an error due to an outstanding loan, and it is borrow
checker's job to compute those loans.  That is, consider *this*
example:

    struct Foo { a: int, b: ~int }
    let x: Foo = ...;
    let y = &x.b;                   //~ NOTE loan issued here
    let z = x.b;                    //~ ERROR move with outstanding loan

In this case, `y` is a pointer into `x`, so when `z` tries to move out
of `x`, we get an error.  There is no way that liveness could compute
this information without redoing the efforts of the borrow checker.

### Closures

Liveness is somewhat complicated by having to deal with stack
closures.  More information to come!

## Distributive property

Copies are "distributive" over parenthesization, but blocks are
considered rvalues.  What this means is that, for example, neither
`a.clone()` nor `(a).clone()` will move `a` (presuming that `a` has a
linear type and `clone()` takes its self by reference), but
`{a}.clone()` will move `a`, as would `(if cond {a} else {b}).clone()`
and so on.

*/

use middle::pat_util::{pat_bindings};
use middle::freevars;
use middle::ty;
use middle::typeck::{method_map};
use util::ppaux;
use util::common::indenter;

use core::hashmap::{HashSet, HashMap};
use syntax::ast::*;
use syntax::ast_util;
use syntax::visit;
use syntax::visit::vt;
use syntax::print::pprust;
use syntax::codemap::span;

#[auto_encode]
#[auto_decode]
pub enum CaptureMode {
    CapCopy, // Copy the value into the closure.
    CapMove, // Move the value into the closure.
    CapRef,  // Reference directly from parent stack frame (used by `&fn()`).
}

#[auto_encode]
#[auto_decode]
pub struct CaptureVar {
    def: def,         // Variable being accessed free
    span: span,       // Location of an access to this variable
    mode: CaptureMode // How variable is being accessed
}

pub type CaptureMap = @mut HashMap<node_id, @[CaptureVar]>;

pub type MovesMap = @mut HashSet<node_id>;

/**
 * For each variable which will be moved, links to the
 * expression */
pub type VariableMovesMap = @mut HashMap<node_id, @expr>;

/** See the section Output on the module comment for explanation. */
pub struct MoveMaps {
    moves_map: MovesMap,
    variable_moves_map: VariableMovesMap,
    capture_map: CaptureMap
}

struct VisitContext {
    tcx: ty::ctxt,
    method_map: method_map,
    move_maps: MoveMaps
}

enum UseMode {
    MoveInWhole,         // Move the entire value.
    MoveInPart(@expr),   // Some subcomponent will be moved
    Read                 // Read no matter what the type.
}

pub fn compute_moves(tcx: ty::ctxt,
                     method_map: method_map,
                     crate: @crate) -> MoveMaps
{
    let visitor = visit::mk_vt(@visit::Visitor {
        visit_expr: compute_modes_for_expr,
        .. *visit::default_visitor()
    });
    let visit_cx = VisitContext {
        tcx: tcx,
        method_map: method_map,
        move_maps: MoveMaps {
            moves_map: @mut HashSet::new(),
            variable_moves_map: @mut HashMap::new(),
            capture_map: @mut HashMap::new()
        }
    };
    visit::visit_crate(crate, visit_cx, visitor);
    return visit_cx.move_maps;
}

// ______________________________________________________________________
// Expressions

fn compute_modes_for_expr(expr: @expr,
                          cx: VisitContext,
                          v: vt<VisitContext>)
{
    cx.consume_expr(expr, v);
}

pub impl UseMode {
    fn component_mode(&self, expr: @expr) -> UseMode {
        /*!
         *
         * Assuming that `self` is the mode for an expression E,
         * returns the appropriate mode to use for a subexpression of E.
         */

        match *self {
            Read | MoveInPart(_) => *self,
            MoveInWhole => MoveInPart(expr)
        }
    }
}

pub impl VisitContext {
    fn consume_exprs(&self,
                     exprs: &[@expr],
                     visitor: vt<VisitContext>)
    {
        for exprs.each |expr| {
            self.consume_expr(*expr, visitor);
        }
    }

    fn consume_expr(&self,
                    expr: @expr,
                    visitor: vt<VisitContext>)
    {
        /*!
         *
         * Indicates that the value of `expr` will be consumed,
         * meaning either copied or moved depending on its type.
         */

        debug!("consume_expr(expr=%?/%s)",
               expr.id,
               pprust::expr_to_str(expr, self.tcx.sess.intr()));

        let expr_ty = ty::expr_ty_adjusted(self.tcx, expr);
        let mode = self.consume_mode_for_ty(expr_ty);
        self.use_expr(expr, mode, visitor);
    }

    fn consume_block(&self,
                     blk: &blk,
                     visitor: vt<VisitContext>)
    {
        /*!
         *
         * Indicates that the value of `blk` will be consumed,
         * meaning either copied or moved depending on its type.
         */

        debug!("consume_block(blk.id=%?)", blk.node.id);

        for blk.node.stmts.each |stmt| {
            (visitor.visit_stmt)(*stmt, *self, visitor);
        }

        for blk.node.expr.each |tail_expr| {
            self.consume_expr(*tail_expr, visitor);
        }
    }

    fn consume_mode_for_ty(&self, ty: ty::t) -> UseMode {
        /*!
         *
         * Selects the appropriate `UseMode` to consume a value with
         * the type `ty`.  This will be `MoveEntireMode` if `ty` is
         * not implicitly copyable.
         */

        let result = if ty::type_moves_by_default(self.tcx, ty) {
            MoveInWhole
        } else {
            Read
        };

        debug!("consume_mode_for_ty(ty=%s) = %?",
               ppaux::ty_to_str(self.tcx, ty), result);

        return result;
    }

    fn use_expr(&self,
                expr: @expr,
                expr_mode: UseMode,
                visitor: vt<VisitContext>)
    {
        /*!
         *
         * Indicates that `expr` is used with a given mode.  This will
         * in turn trigger calls to the subcomponents of `expr`.
         */

        debug!("use_expr(expr=%?/%s, mode=%?)",
               expr.id, pprust::expr_to_str(expr, self.tcx.sess.intr()),
               expr_mode);

        match expr_mode {
            MoveInWhole => { self.move_maps.moves_map.insert(expr.id); }
            MoveInPart(_) | Read => {}
        }

        // `expr_mode` refers to the post-adjustment value.  If one of
        // those adjustments is to take a reference, then it's only
        // reading the underlying expression, not moving it.
        let comp_mode = match self.tcx.adjustments.find(&expr.id) {
            Some(&@ty::AutoDerefRef(
                ty::AutoDerefRef {
                    autoref: Some(_), _})) => Read,
            _ => expr_mode.component_mode(expr)
        };

        debug!("comp_mode = %?", comp_mode);

        match expr.node {
            expr_path(*) => {
                match comp_mode {
                    MoveInPart(entire_expr) => {
                        self.move_maps.variable_moves_map.insert(
                            expr.id, entire_expr);
                    }
                    Read => {}
                    MoveInWhole => {
                        self.tcx.sess.span_bug(
                            expr.span,
                            fmt!("Component mode can never be MoveInWhole"));
                    }
                }
            }

            expr_unary(deref, base) => {       // *base
                if !self.use_overloaded_operator(
                    expr, base, [], visitor)
                {
                    // Moving out of *base moves out of base.
                    self.use_expr(base, comp_mode, visitor);
                }
            }

            expr_field(base, _, _) => {        // base.f
                // Moving out of base.f moves out of base.
                self.use_expr(base, comp_mode, visitor);
            }

            expr_index(lhs, rhs) => {          // lhs[rhs]
                if !self.use_overloaded_operator(
                    expr, lhs, [rhs], visitor)
                {
                    self.use_expr(lhs, comp_mode, visitor);
                    self.consume_expr(rhs, visitor);
                }
            }

            expr_call(callee, ref args, _) => {    // callee(args)
                self.use_expr(callee, Read, visitor);
                self.use_fn_args(callee.id, *args, visitor);
            }

            expr_method_call(callee, _, _, ref args, _) => { // callee.m(args)
                // Implicit self is equivalent to & mode, but every
                // other kind should be + mode.
                self.use_receiver(callee, visitor);
                self.use_fn_args(expr.callee_id, *args, visitor);
            }

            expr_struct(_, ref fields, opt_with) => {
                for fields.each |field| {
                    self.consume_expr(field.node.expr, visitor);
                }

                for opt_with.each |with_expr| {
                    // If there are any fields whose type is move-by-default,
                    // then `with` is consumed, otherwise it is only read
                    let with_ty = ty::expr_ty(self.tcx, *with_expr);
                    let with_fields = match ty::get(with_ty).sty {
                        ty::ty_struct(did, ref substs) => {
                            ty::struct_fields(self.tcx, did, substs)
                        }
                        ref r => {
                           self.tcx.sess.span_bug(
                                with_expr.span,
                                fmt!("bad base expr type in record: %?", r))
                        }
                    };

                    // The `with` expr must be consumed if it contains
                    // any fields which (1) were not explicitly
                    // specified and (2) have a type that
                    // moves-by-default:
                    let consume_with = with_fields.any(|tf| {
                        !fields.any(|f| f.node.ident == tf.ident) &&
                            ty::type_moves_by_default(self.tcx, tf.mt.ty)
                    });

                    if consume_with {
                        self.consume_expr(*with_expr, visitor);
                    } else {
                        self.use_expr(*with_expr, Read, visitor);
                    }
                }
            }

            expr_tup(ref exprs) => {
                self.consume_exprs(*exprs, visitor);
            }

            expr_if(cond_expr, ref then_blk, opt_else_expr) => {
                self.consume_expr(cond_expr, visitor);
                self.consume_block(then_blk, visitor);
                for opt_else_expr.each |else_expr| {
                    self.consume_expr(*else_expr, visitor);
                }
            }

            expr_match(discr, ref arms) => {
                // We must do this first so that `arms_have_by_move_bindings`
                // below knows which bindings are moves.
                for arms.each |arm| {
                    self.consume_arm(arm, visitor);
                }

                let by_move_bindings_present =
                    self.arms_have_by_move_bindings(
                        self.move_maps.moves_map, *arms);

                if by_move_bindings_present {
                    // If one of the arms moves a value out of the
                    // discriminant, then the discriminant itself is
                    // moved.
                    self.consume_expr(discr, visitor);
                } else {
                    // Otherwise, the discriminant is merely read.
                    self.use_expr(discr, Read, visitor);
                }
            }

            expr_copy(base) => {
                self.use_expr(base, Read, visitor);
            }

            expr_paren(base) => {
                // Note: base is not considered a *component* here, so
                // use `expr_mode` not `comp_mode`.
                self.use_expr(base, expr_mode, visitor);
            }

            expr_vec(ref exprs, _) => {
                self.consume_exprs(*exprs, visitor);
            }

            expr_addr_of(_, base) => {   // &base
                self.use_expr(base, Read, visitor);
            }

            expr_inline_asm(*) |
            expr_break(*) |
            expr_again(*) |
            expr_lit(*) => {}

            expr_loop(ref blk, _) => {
                self.consume_block(blk, visitor);
            }

            expr_log(a_expr, b_expr) => {
                self.consume_expr(a_expr, visitor);
                self.use_expr(b_expr, Read, visitor);
            }

            expr_while(cond_expr, ref blk) => {
                self.consume_expr(cond_expr, visitor);
                self.consume_block(blk, visitor);
            }

            expr_unary(_, lhs) => {
                if !self.use_overloaded_operator(
                    expr, lhs, [], visitor)
                {
                    self.consume_expr(lhs, visitor);
                }
            }

            expr_binary(_, lhs, rhs) => {
                if !self.use_overloaded_operator(
                    expr, lhs, [rhs], visitor)
                {
                    self.consume_expr(lhs, visitor);
                    self.consume_expr(rhs, visitor);
                }
            }

            expr_block(ref blk) => {
                self.consume_block(blk, visitor);
            }

            expr_ret(ref opt_expr) => {
                for opt_expr.each |expr| {
                    self.consume_expr(*expr, visitor);
                }
            }

            expr_assign(lhs, rhs) => {
                self.use_expr(lhs, Read, visitor);
                self.consume_expr(rhs, visitor);
            }

            expr_cast(base, _) => {
                self.consume_expr(base, visitor);
            }

            expr_assign_op(_, lhs, rhs) => {
                // FIXME(#4712) --- Overloaded operators?
                //
                // if !self.use_overloaded_operator(
                //     expr, DoDerefArgs, lhs, [rhs], visitor)
                // {
                self.consume_expr(lhs, visitor);
                self.consume_expr(rhs, visitor);
                // }
            }

            expr_repeat(base, count, _) => {
                self.consume_expr(base, visitor);
                self.consume_expr(count, visitor);
            }

            expr_swap(lhs, rhs) => {
                self.use_expr(lhs, Read, visitor);
                self.use_expr(rhs, Read, visitor);
            }

            expr_loop_body(base) |
            expr_do_body(base) => {
                self.use_expr(base, comp_mode, visitor);
            }

            expr_fn_block(_, ref body) => {
                let cap_vars = self.compute_captures(expr.id);
                self.move_maps.capture_map.insert(expr.id, cap_vars);
                self.consume_block(body, visitor);
            }

            expr_vstore(base, _) => {
                self.use_expr(base, comp_mode, visitor);
            }

            expr_mac(*) => {
                self.tcx.sess.span_bug(
                    expr.span,
                    ~"macro expression remains after expansion");
            }
        }
    }

    fn use_overloaded_operator(&self,
                               expr: @expr,
                               receiver_expr: @expr,
                               arg_exprs: &[@expr],
                               visitor: vt<VisitContext>) -> bool
    {
        if !self.method_map.contains_key(&expr.id) {
            return false;
        }

        self.use_receiver(receiver_expr, visitor);

        // for overloaded operatrs, we are always passing in a
        // borrowed pointer, so it's always read mode:
        for arg_exprs.each |arg_expr| {
            self.use_expr(*arg_expr, Read, visitor);
        }

        return true;
    }

    fn consume_arm(&self,
                   arm: &arm,
                   visitor: vt<VisitContext>)
    {
        for arm.pats.each |pat| {
            self.use_pat(*pat);
        }

        for arm.guard.each |guard| {
            self.consume_expr(*guard, visitor);
        }

        self.consume_block(&arm.body, visitor);
    }

    fn use_pat(&self,
               pat: @pat)
    {
        /*!
         *
         * Decides whether each binding in a pattern moves the value
         * into itself or not based on its type and annotation.
         */

        do pat_bindings(self.tcx.def_map, pat) |bm, id, _span, _path| {
            let mode = match bm {
                bind_by_copy => Read,
                bind_by_ref(_) => Read,
                bind_infer => {
                    let pat_ty = ty::node_id_to_type(self.tcx, id);
                    self.consume_mode_for_ty(pat_ty)
                }
            };

            match mode {
                MoveInWhole => { self.move_maps.moves_map.insert(id); }
                MoveInPart(_) | Read => {}
            }
        }
    }

    fn use_receiver(&self,
                    receiver_expr: @expr,
                    visitor: vt<VisitContext>)
    {
        self.use_fn_arg(by_copy, receiver_expr, visitor);
    }

    fn use_fn_args(&self,
                   callee_id: node_id,
                   arg_exprs: &[@expr],
                   visitor: vt<VisitContext>)
    {
        /*!
         *
         * Uses the argument expressions according to the function modes.
         */

        let arg_tys =
            ty::ty_fn_args(ty::node_id_to_type(self.tcx, callee_id));
        for vec::each2(arg_exprs, arg_tys) |arg_expr, arg_ty| {
            let arg_mode = ty::resolved_mode(self.tcx, arg_ty.mode);
            self.use_fn_arg(arg_mode, *arg_expr, visitor);
        }
    }

    fn use_fn_arg(&self,
                  arg_mode: rmode,
                  arg_expr: @expr,
                  visitor: vt<VisitContext>)
    {
        /*!
         *
         * Uses the argument according to the given argument mode.
         */

        match arg_mode {
            by_ref => self.use_expr(arg_expr, Read, visitor),
            by_copy => self.consume_expr(arg_expr, visitor)
        }
    }

    fn arms_have_by_move_bindings(&self,
                                  moves_map: MovesMap,
                                  arms: &[arm]) -> bool
    {
        for arms.each |arm| {
            for arm.pats.each |pat| {
                let mut found = false;
                do pat_bindings(self.tcx.def_map, *pat) |_, node_id, _, _| {
                    if moves_map.contains(&node_id) {
                        found = true;
                    }
                }
                if found { return true; }
            }
        }
        return false;
    }

    fn compute_captures(&self, fn_expr_id: node_id) -> @[CaptureVar] {
        debug!("compute_capture_vars(fn_expr_id=%?)", fn_expr_id);
        let _indenter = indenter();

        let fn_ty = ty::node_id_to_type(self.tcx, fn_expr_id);
        let sigil = ty::ty_closure_sigil(fn_ty);
        let freevars = freevars::get_freevars(self.tcx, fn_expr_id);
        if sigil == BorrowedSigil {
            // &fn() captures everything by ref
            at_vec::from_fn(freevars.len(), |i| {
                let fvar = &freevars[i];
                CaptureVar {def: fvar.def, span: fvar.span, mode: CapRef}
            })
        } else {
            // @fn() and ~fn() capture by copy or by move depending on type
            at_vec::from_fn(freevars.len(), |i| {
                let fvar = &freevars[i];
                let fvar_def_id = ast_util::def_id_of_def(fvar.def).node;
                let fvar_ty = ty::node_id_to_type(self.tcx, fvar_def_id);
                debug!("fvar_def_id=%? fvar_ty=%s",
                       fvar_def_id, ppaux::ty_to_str(self.tcx, fvar_ty));
                let mode = if ty::type_moves_by_default(self.tcx, fvar_ty) {
                    CapMove
                } else {
                    CapCopy
                };
                CaptureVar {def: fvar.def, span: fvar.span, mode:mode}
            })
        }
    }
}

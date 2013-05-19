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
 * A module for propagating forward dataflow information. The analysis
 * assumes that the items to be propagated can be represented as bits
 * and thus uses bitvectors. Your job is simply to specify the so-called
 * GEN and KILL bits for each expression.
 */

use core::cast;
use core::uint;
use syntax::ast;
use syntax::ast_util;
use syntax::ast_util::id_range;
use syntax::print::{pp, pprust};
use middle::ty;
use middle::typeck;
use util::ppaux::Repr;

pub struct DataFlowContext<O> {
    priv tcx: ty::ctxt,
    priv method_map: typeck::method_map,

    /// the data flow operator
    priv oper: O,

    /// range of ids that appear within the item in question
    priv id_range: id_range,

    /// number of bits to propagate per id
    priv bits_per_id: uint,

    /// number of words we will use to store bits_per_id.
    /// equal to bits_per_id/uint::bits rounded up.
    priv words_per_id: uint,

    // Bit sets per id.  The following three fields (`gens`, `kills`,
    // and `on_entry`) all have the same structure. For each id in
    // `id_range`, there is a range of words equal to `words_per_id`.
    // So, to access the bits for any given id, you take a slice of
    // the full vector (see the method `compute_id_range()`).

    /// bits generated as we exit the scope `id`. Updated by `add_gen()`.
    priv gens: ~[uint],

    /// bits killed as we exit the scope `id`. Updated by `add_kill()`.
    priv kills: ~[uint],

    /// bits that are valid on entry to the scope `id`. Updated by
    /// `propagate()`.
    priv on_entry: ~[uint]
}

/// Parameterization for the precise form of data flow that is used.
pub trait DataFlowOperator {
    /// Specifies the initial value for each bit in the `on_entry` set
    fn initial_value(&self) -> bool;

    /// Joins two predecessor bits together, typically either `|` or `&`
    fn join(&self, succ: uint, pred: uint) -> uint;

    /// True if we should propagate through closures
    fn walk_closures(&self) -> bool;
}

struct PropagationContext<'self, O> {
    dfcx: &'self mut DataFlowContext<O>,
    changed: bool
}

#[deriving(Eq)]
enum LoopKind {
    /// A `while` or `loop` loop
    TrueLoop,

    /// A `for` "loop" (i.e., really a func call where `break`, `return`,
    /// and `loop` all essentially perform an early return from the closure)
    ForLoop
}

struct LoopScope<'self> {
    loop_id: ast::node_id,
    loop_kind: LoopKind,
    break_bits: ~[uint]
}

impl<O:DataFlowOperator> DataFlowContext<O> {
    pub fn new(tcx: ty::ctxt,
               method_map: typeck::method_map,
               oper: O,
               id_range: id_range,
               bits_per_id: uint) -> DataFlowContext<O> {
        let words_per_id = (bits_per_id + uint::bits - 1) / uint::bits;

        debug!("DataFlowContext::new(id_range=%?, bits_per_id=%?, words_per_id=%?)",
               id_range, bits_per_id, words_per_id);

        let len = (id_range.max - id_range.min) as uint * words_per_id;
        let gens = vec::from_elem(len, 0);
        let kills = vec::from_elem(len, 0);
        let elem = if oper.initial_value() {uint::max_value} else {0};
        let on_entry = vec::from_elem(len, elem);

        DataFlowContext {
            tcx: tcx,
            method_map: method_map,
            words_per_id: words_per_id,
            bits_per_id: bits_per_id,
            oper: oper,
            id_range: id_range,
            gens: gens,
            kills: kills,
            on_entry: on_entry
        }
    }

    pub fn add_gen(&mut self, id: ast::node_id, bit: uint) {
        //! Indicates that `id` generates `bit`

        debug!("add_gen(id=%?, bit=%?)", id, bit);
        let (start, end) = self.compute_id_range(id);
        {
            let gens = vec::mut_slice(self.gens, start, end);
            set_bit(gens, bit);
        }
    }

    pub fn add_kill(&mut self, id: ast::node_id, bit: uint) {
        //! Indicates that `id` kills `bit`

        debug!("add_kill(id=%?, bit=%?)", id, bit);
        let (start, end) = self.compute_id_range(id);
        {
            let kills = vec::mut_slice(self.kills, start, end);
            set_bit(kills, bit);
        }
    }

    fn apply_gen_kill(&self, id: ast::node_id, bits: &mut [uint]) {
        //! Applies the gen and kill sets for `id` to `bits`

        debug!("apply_gen_kill(id=%?, bits=%s) [before]",
               id, mut_bits_to_str(bits));
        let (start, end) = self.compute_id_range(id);
        let gens = self.gens.slice(start, end);
        bitwise(bits, gens, |a, b| a | b);
        let kills = self.kills.slice(start, end);
        bitwise(bits, kills, |a, b| a & !b);

        debug!("apply_gen_kill(id=%?, bits=%s) [after]",
               id, mut_bits_to_str(bits));
    }

    fn apply_kill(&self, id: ast::node_id, bits: &mut [uint]) {
        debug!("apply_kill(id=%?, bits=%s) [before]",
               id, mut_bits_to_str(bits));
        let (start, end) = self.compute_id_range(id);
        let kills = self.kills.slice(start, end);
        bitwise(bits, kills, |a, b| a & !b);
        debug!("apply_kill(id=%?, bits=%s) [after]",
               id, mut_bits_to_str(bits));
    }

    fn compute_id_range(&self, absolute_id: ast::node_id) -> (uint, uint) {
        assert!(absolute_id >= self.id_range.min);
        assert!(absolute_id < self.id_range.max);

        let relative_id = absolute_id - self.id_range.min;
        let start = (relative_id as uint) * self.words_per_id;
        let end = start + self.words_per_id;
        (start, end)
    }


    pub fn each_bit_on_entry(&self,
                             id: ast::node_id,
                             f: &fn(uint) -> bool) -> bool {
        //! Iterates through each bit that is set on entry to `id`.
        //! Only useful after `propagate()` has been called.

        let (start, end) = self.compute_id_range(id);
        let on_entry = vec::slice(self.on_entry, start, end);
        debug!("each_bit_on_entry(id=%?, on_entry=%s)",
               id, bits_to_str(on_entry));
        self.each_bit(on_entry, f)
    }

    pub fn each_gen_bit(&self,
                        id: ast::node_id,
                        f: &fn(uint) -> bool) -> bool {
        //! Iterates through each bit in the gen set for `id`.

        let (start, end) = self.compute_id_range(id);
        let gens = vec::slice(self.gens, start, end);
        debug!("each_gen_bit(id=%?, gens=%s)",
               id, bits_to_str(gens));
        self.each_bit(gens, f)
    }

    fn each_bit(&self,
                words: &[uint],
                f: &fn(uint) -> bool) -> bool {
        //! Helper for iterating over the bits in a bit set.

        for words.eachi |word_index, &word| {
            if word != 0 {
                let base_index = word_index * uint::bits;
                for uint::range(0, uint::bits) |offset| {
                    let bit = 1 << offset;
                    if (word & bit) != 0 {
                        // NB: we round up the total number of bits
                        // that we store in any given bit set so that
                        // it is an even multiple of uint::bits.  This
                        // means that there may be some stray bits at
                        // the end that do not correspond to any
                        // actual value.  So before we callback, check
                        // whether the bit_index is greater than the
                        // actual value the user specified and stop
                        // iterating if so.
                        let bit_index = base_index + offset;
                        if bit_index >= self.bits_per_id {
                            return true;
                        } else if !f(bit_index) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
}

impl<O:DataFlowOperator+Copy+'static> DataFlowContext<O> {
//                      ^^^^^^^^^^^^ only needed for pretty printing
    pub fn propagate(&mut self, blk: &ast::blk) {
        //! Performs the data flow analysis.

        if self.bits_per_id == 0 {
            // Optimize the surprisingly common degenerate case.
            return;
        }

        let mut propcx = PropagationContext {
            dfcx: self,
            changed: true
        };

        let mut temp = vec::from_elem(self.words_per_id, 0);
        let mut loop_scopes = ~[];

        while propcx.changed {
            propcx.changed = false;
            propcx.reset(temp);
            propcx.walk_block(blk, temp, &mut loop_scopes);
        }

        debug!("Dataflow result:");
        debug!("%s", {
            let this = @copy *self;
            this.pretty_print_to(io::stderr(), blk);
            ""
        });
    }

    fn pretty_print_to(@self, wr: @io::Writer, blk: &ast::blk) {
        let pre: @fn(pprust::ann_node) = |node| {
            let (ps, id) = match node {
                pprust::node_expr(ps, expr) => (ps, expr.id),
                pprust::node_block(ps, blk) => (ps, blk.node.id),
                pprust::node_item(ps, _) => (ps, 0),
                pprust::node_pat(ps, pat) => (ps, pat.id)
            };

            if id >= self.id_range.min || id < self.id_range.max {
                let (start, end) = self.compute_id_range(id);
                let on_entry = vec::slice(self.on_entry, start, end);
                let entry_str = bits_to_str(on_entry);

                let gens = vec::slice(self.gens, start, end);
                let gens_str = if gens.any(|&u| u != 0) {
                    fmt!(" gen: %s", bits_to_str(gens))
                } else {
                    ~""
                };

                let kills = vec::slice(self.kills, start, end);
                let kills_str = if kills.any(|&u| u != 0) {
                    fmt!(" kill: %s", bits_to_str(kills))
                } else {
                    ~""
                };

                let comment_str = fmt!("id %d: %s%s%s",
                                       id, entry_str, gens_str, kills_str);
                pprust::synth_comment(ps, comment_str);
                pp::space(ps.s);
            }
        };

        let post: @fn(pprust::ann_node) = |_| {
        };

        let ps = pprust::rust_printer_annotated(
            wr, self.tcx.sess.intr(),
            pprust::pp_ann {pre:pre, post:post});
        pprust::cbox(ps, pprust::indent_unit);
        pprust::ibox(ps, 0u);
        pprust::print_block(ps, blk);
        pp::eof(ps.s);
    }
}

impl<'self, O:DataFlowOperator> PropagationContext<'self, O> {
    fn tcx(&self) -> ty::ctxt {
        self.dfcx.tcx
    }

    fn walk_block(&mut self,
                  blk: &ast::blk,
                  in_out: &mut [uint],
                  loop_scopes: &mut ~[LoopScope]) {
        debug!("DataFlowContext::walk_block(blk.node.id=%?, in_out=%s)",
               blk.node.id, bits_to_str(reslice(in_out)));

        self.merge_with_entry_set(blk.node.id, in_out);

        for blk.node.stmts.each |&stmt| {
            self.walk_stmt(stmt, in_out, loop_scopes);
        }

        self.walk_opt_expr(blk.node.expr, in_out, loop_scopes);

        self.dfcx.apply_gen_kill(blk.node.id, in_out);
    }

    fn walk_stmt(&mut self,
                 stmt: @ast::stmt,
                 in_out: &mut [uint],
                 loop_scopes: &mut ~[LoopScope]) {
        match stmt.node {
            ast::stmt_decl(decl, _) => {
                self.walk_decl(decl, in_out, loop_scopes);
            }

            ast::stmt_expr(expr, _) | ast::stmt_semi(expr, _) => {
                self.walk_expr(expr, in_out, loop_scopes);
            }

            ast::stmt_mac(*) => {
                self.tcx().sess.span_bug(stmt.span, "unexpanded macro");
            }
        }
    }

    fn walk_decl(&mut self,
                 decl: @ast::decl,
                 in_out: &mut [uint],
                 loop_scopes: &mut ~[LoopScope]) {
        match decl.node {
            ast::decl_local(ref locals) => {
                for locals.each |local| {
                    self.walk_pat(local.node.pat, in_out, loop_scopes);
                    self.walk_opt_expr(local.node.init, in_out, loop_scopes);
                }
            }

            ast::decl_item(_) => {}
        }
    }

    fn walk_expr(&mut self,
                 expr: @ast::expr,
                 in_out: &mut [uint],
                 loop_scopes: &mut ~[LoopScope]) {
        debug!("DataFlowContext::walk_expr(expr=%s, in_out=%s)",
               expr.repr(self.dfcx.tcx), bits_to_str(reslice(in_out)));

        self.merge_with_entry_set(expr.id, in_out);

        match expr.node {
            ast::expr_fn_block(ref decl, ref body) => {
                if self.dfcx.oper.walk_closures() {
                    // In the absence of once fns, we must assume that
                    // every function body will execute more than
                    // once. Thus we treat every function body like a
                    // loop.
                    //
                    // What is subtle and a bit tricky, also, is how
                    // to deal with the "output" bits---that is, what
                    // do we consider to be the successor of a
                    // function body, given that it could be called
                    // from any point within its lifetime? What we do
                    // is to add their effects immediately as of the
                    // point of creation. Of course we have to ensure
                    // that this is sound for the analyses which make
                    // use of dataflow.
                    //
                    // In the case of the initedness checker (which
                    // does not currently use dataflow, but I hope to
                    // convert at some point), we will simply not walk
                    // closures at all, so it's a moot point.
                    //
                    // In the case of the borrow checker, this means
                    // the loans which would be created by calling a
                    // function come into effect immediately when the
                    // function is created. This is guaranteed to be
                    // earlier than the point at which the loan
                    // actually comes into scope (which is the point
                    // at which the closure is *called*). Because
                    // loans persist until the scope of the loans is
                    // exited, it is always a safe approximation to
                    // have a loan begin earlier than it actually will
                    // at runtime, so this should be sound.
                    //
                    // We stil have to be careful in the region
                    // checker and borrow checker to treat function
                    // bodies like loops, which implies some
                    // limitations. For example, a closure cannot root
                    // a managed box for longer than its body.
                    //
                    // General control flow looks like this:
                    //
                    //  +- (expr) <----------+
                    //  |    |               |
                    //  |    v               |
                    //  |  (body) -----------+--> (exit)
                    //  |    |               |
                    //  |    + (break/loop) -+
                    //  |                    |
                    //  +--------------------+
                    //
                    // This is a bit more conservative than a loop.
                    // Note that we must assume that even after a
                    // `break` occurs (e.g., in a `for` loop) that the
                    // closure may be reinvoked.
                    //
                    // One difference from other loops is that `loop`
                    // and `break` statements which target a closure
                    // both simply add to the `break_bits`.

                    // func_bits represents the state when the function
                    // returns
                    let mut func_bits = reslice(in_out).to_vec();

                    loop_scopes.push(LoopScope {
                        loop_id: expr.id,
                        loop_kind: ForLoop,
                        break_bits: reslice(in_out).to_vec()
                    });
                    for decl.inputs.each |input| {
                        self.walk_pat(input.pat, func_bits, loop_scopes);
                    }
                    self.walk_block(body, func_bits, loop_scopes);

                    // add the bits from any early return via `break`,
                    // `continue`, or `return` into `func_bits`
                    let loop_scope = loop_scopes.pop();
                    join_bits(&self.dfcx.oper, loop_scope.break_bits, func_bits);

                    // add `func_bits` to the entry bits for `expr`,
                    // since we must assume the function may be called
                    // more than once
                    self.add_to_entry_set(expr.id, reslice(func_bits));

                    // the final exit bits include whatever was present
                    // in the original, joined with the bits from the function
                    join_bits(&self.dfcx.oper, func_bits, in_out);
                }
            }

            ast::expr_if(cond, ref then, els) => {
                //
                //     (cond)
                //       |
                //       v
                //      ( )
                //     /   \
                //    |     |
                //    v     v
                //  (then)(els)
                //    |     |
                //    v     v
                //   (  succ  )
                //
                self.walk_expr(cond, in_out, loop_scopes);

                let mut then_bits = reslice(in_out).to_vec();
                self.walk_block(then, then_bits, loop_scopes);

                self.walk_opt_expr(els, in_out, loop_scopes);
                join_bits(&self.dfcx.oper, then_bits, in_out);
            }

            ast::expr_while(cond, ref blk) => {
                //
                //     (expr) <--+
                //       |       |
                //       v       |
                //  +--(cond)    |
                //  |    |       |
                //  |    v       |
                //  v  (blk) ----+
                //       |
                //    <--+ (break)
                //

                self.walk_expr(cond, in_out, loop_scopes);

                let mut body_bits = reslice(in_out).to_vec();
                loop_scopes.push(LoopScope {
                    loop_id: expr.id,
                    loop_kind: TrueLoop,
                    break_bits: reslice(in_out).to_vec()
                });
                self.walk_block(blk, body_bits, loop_scopes);
                self.add_to_entry_set(expr.id, body_bits);
                let new_loop_scope = loop_scopes.pop();
                copy_bits(new_loop_scope.break_bits, in_out);
            }

            ast::expr_loop(ref blk, _) => {
                //
                //     (expr) <--+
                //       |       |
                //       v       |
                //     (blk) ----+
                //       |
                //    <--+ (break)
                //

                let mut body_bits = reslice(in_out).to_vec();
                self.reset(in_out);
                loop_scopes.push(LoopScope {
                    loop_id: expr.id,
                    loop_kind: TrueLoop,
                    break_bits: reslice(in_out).to_vec()
                });
                self.walk_block(blk, body_bits, loop_scopes);
                self.add_to_entry_set(expr.id, body_bits);

                let new_loop_scope = loop_scopes.pop();
                assert_eq!(new_loop_scope.loop_id, expr.id);
                copy_bits(new_loop_scope.break_bits, in_out);
            }

            ast::expr_match(discr, ref arms) => {
                //
                //    (discr)
                //     / | \
                //    |  |  |
                //    v  v  v
                //   (..arms..)
                //    |  |  |
                //    v  v  v
                //   (  succ  )
                //
                //
                self.walk_expr(discr, in_out, loop_scopes);

                let mut guards = reslice(in_out).to_vec();

                // We know that exactly one arm will be taken, so we
                // can start out with a blank slate and just union
                // together the bits from each arm:
                self.reset(in_out);

                for arms.each |arm| {
                    // in_out reflects the discr and all guards to date
                    self.walk_opt_expr(arm.guard, guards, loop_scopes);

                    // determine the bits for the body and then union
                    // them into `in_out`, which reflects all bodies to date
                    let mut body = reslice(guards).to_vec();
                    self.walk_pat_alternatives(arm.pats, body, loop_scopes);
                    self.walk_block(&arm.body, body, loop_scopes);
                    join_bits(&self.dfcx.oper, body, in_out);
                }
            }

            ast::expr_ret(o_e) => {
                self.walk_opt_expr(o_e, in_out, loop_scopes);

                // is this a return from a `for`-loop closure?
                match loop_scopes.position(|s| s.loop_kind == ForLoop) {
                    Some(i) => {
                        // if so, add the in_out bits to the state
                        // upon exit. Remember that we cannot count
                        // upon the `for` loop function not to invoke
                        // the closure again etc.
                        self.break_from_to(expr, &mut loop_scopes[i], in_out);
                    }

                    None => {}
                }

                self.reset(in_out);
            }

            ast::expr_break(label) => {
                let scope = self.find_scope(expr, label, loop_scopes);
                self.break_from_to(expr, scope, in_out);
                self.reset(in_out);
            }

            ast::expr_again(label) => {
                let scope = self.find_scope(expr, label, loop_scopes);

                match scope.loop_kind {
                    TrueLoop => {
                        self.pop_scopes(expr, scope, in_out);
                        self.add_to_entry_set(scope.loop_id, reslice(in_out));
                    }

                    ForLoop => {
                        // If this `loop` construct is looping back to a `for`
                        // loop, then `loop` is really just a return from the
                        // closure. Therefore, we treat it the same as `break`.
                        // See case for `expr_fn_block` for more details.
                        self.break_from_to(expr, scope, in_out);
                    }
                }

                self.reset(in_out);
            }

            ast::expr_assign(l, r) |
            ast::expr_assign_op(_, l, r) => {
                self.walk_expr(r, in_out, loop_scopes);
                self.walk_expr(l, in_out, loop_scopes);
            }

            ast::expr_vec(ref exprs, _) => {
                self.walk_exprs(*exprs, in_out, loop_scopes)
            }

            ast::expr_repeat(l, r, _) => {
                self.walk_expr(l, in_out, loop_scopes);
                self.walk_expr(r, in_out, loop_scopes);
            }

            ast::expr_struct(_, ref fields, with_expr) => {
                self.walk_opt_expr(with_expr, in_out, loop_scopes);
                for fields.each |field| {
                    self.walk_expr(field.node.expr, in_out, loop_scopes);
                }
            }

            ast::expr_call(f, ref args, _) => {
                self.walk_call(expr.callee_id, expr.id,
                               f, *args, in_out, loop_scopes);
            }

            ast::expr_method_call(rcvr, _, _, ref args, _) => {
                self.walk_call(expr.callee_id, expr.id,
                               rcvr, *args, in_out, loop_scopes);
            }

            ast::expr_index(l, r) |
            ast::expr_binary(_, l, r) if self.is_method_call(expr) => {
                self.walk_call(expr.callee_id, expr.id,
                               l, [r], in_out, loop_scopes);
            }

            ast::expr_unary(_, e) if self.is_method_call(expr) => {
                self.walk_call(expr.callee_id, expr.id,
                               e, [], in_out, loop_scopes);
            }

            ast::expr_tup(ref exprs) => {
                self.walk_exprs(*exprs, in_out, loop_scopes);
            }

            ast::expr_binary(op, l, r) if ast_util::lazy_binop(op) => {
                self.walk_expr(l, in_out, loop_scopes);
                let temp = reslice(in_out).to_vec();
                self.walk_expr(r, in_out, loop_scopes);
                join_bits(&self.dfcx.oper, temp, in_out);
            }

            ast::expr_log(l, r) |
            ast::expr_index(l, r) |
            ast::expr_binary(_, l, r) => {
                self.walk_exprs([l, r], in_out, loop_scopes);
            }

            ast::expr_lit(*) |
            ast::expr_path(*) |
            ast::expr_self => {
            }

            ast::expr_addr_of(_, e) |
            ast::expr_copy(e) |
            ast::expr_loop_body(e) |
            ast::expr_do_body(e) |
            ast::expr_cast(e, _) |
            ast::expr_unary(_, e) |
            ast::expr_paren(e) |
            ast::expr_vstore(e, _) |
            ast::expr_field(e, _, _) => {
                self.walk_expr(e, in_out, loop_scopes);
            }

            ast::expr_inline_asm(ref inline_asm) => {
                for inline_asm.inputs.each |&(_, expr)| {
                    self.walk_expr(expr, in_out, loop_scopes);
                }
                for inline_asm.outputs.each |&(_, expr)| {
                    self.walk_expr(expr, in_out, loop_scopes);
                }
            }

            ast::expr_block(ref blk) => {
                self.walk_block(blk, in_out, loop_scopes);
            }

            ast::expr_mac(*) => {
                self.tcx().sess.span_bug(expr.span, "unexpanded macro");
            }
        }

        self.dfcx.apply_gen_kill(expr.id, in_out);
    }

    fn pop_scopes(&mut self,
                  from_expr: @ast::expr,
                  to_scope: &mut LoopScope,
                  in_out: &mut [uint]) {
        //! Whenever you have a `break` or a `loop` statement, flow
        //! exits through any number of enclosing scopes on its
        //! way to the new destination. This function applies the kill
        //! sets of those enclosing scopes to `in_out` (those kill sets
        //! concern items that are going out of scope).

        let tcx = self.tcx();
        let region_maps = tcx.region_maps;

        debug!("pop_scopes(from_expr=%s, to_scope=%?, in_out=%s)",
               from_expr.repr(tcx), to_scope.loop_id,
               bits_to_str(reslice(in_out)));

        let mut id = from_expr.id;
        while id != to_scope.loop_id {
            self.dfcx.apply_kill(id, in_out);

            match region_maps.opt_encl_scope(id) {
                Some(i) => { id = i; }
                None => {
                    tcx.sess.span_bug(
                        from_expr.span,
                        fmt!("pop_scopes(from_expr=%s, to_scope=%?) \
                              to_scope does not enclose from_expr",
                             from_expr.repr(tcx), to_scope.loop_id));
                }
            }
        }
    }

    fn break_from_to(&mut self,
                     from_expr: @ast::expr,
                     to_scope: &mut LoopScope,
                     in_out: &mut [uint]) {
        self.pop_scopes(from_expr, to_scope, in_out);
        self.dfcx.apply_kill(from_expr.id, in_out);
        join_bits(&self.dfcx.oper, reslice(in_out), to_scope.break_bits);
        debug!("break_from_to(from_expr=%s, to_scope=%?) final break_bits=%s",
               from_expr.repr(self.tcx()),
               to_scope.loop_id,
               bits_to_str(reslice(in_out)));
    }

    fn walk_exprs(&mut self,
                  exprs: &[@ast::expr],
                  in_out: &mut [uint],
                  loop_scopes: &mut ~[LoopScope]) {
        for exprs.each |&expr| {
            self.walk_expr(expr, in_out, loop_scopes);
        }
    }

    fn walk_opt_expr(&mut self,
                     opt_expr: Option<@ast::expr>,
                     in_out: &mut [uint],
                     loop_scopes: &mut ~[LoopScope]) {
        for opt_expr.each |&expr| {
            self.walk_expr(expr, in_out, loop_scopes);
        }
    }

    fn walk_call(&mut self,
                 _callee_id: ast::node_id,
                 call_id: ast::node_id,
                 arg0: @ast::expr,
                 args: &[@ast::expr],
                 in_out: &mut [uint],
                 loop_scopes: &mut ~[LoopScope]) {
        self.walk_expr(arg0, in_out, loop_scopes);
        self.walk_exprs(args, in_out, loop_scopes);

        // FIXME(#6268) nested method calls
        // self.merge_with_entry_set(callee_id, in_out);
        // self.dfcx.apply_gen_kill(callee_id, in_out);

        let return_ty = ty::node_id_to_type(self.tcx(), call_id);
        let fails = ty::type_is_bot(return_ty);
        if fails {
            self.reset(in_out);
        }
    }

    fn walk_pat(&mut self,
                pat: @ast::pat,
                in_out: &mut [uint],
                _loop_scopes: &mut ~[LoopScope]) {
        debug!("DataFlowContext::walk_pat(pat=%s, in_out=%s)",
               pat.repr(self.dfcx.tcx), bits_to_str(reslice(in_out)));

        do ast_util::walk_pat(pat) |p| {
            debug!("  p.id=%? in_out=%s", p.id, bits_to_str(reslice(in_out)));
            self.merge_with_entry_set(p.id, in_out);
            self.dfcx.apply_gen_kill(p.id, in_out);
        }
    }

    fn walk_pat_alternatives(&mut self,
                             pats: &[@ast::pat],
                             in_out: &mut [uint],
                             loop_scopes: &mut ~[LoopScope]) {
        if pats.len() == 1 {
            // Common special case:
            return self.walk_pat(pats[0], in_out, loop_scopes);
        }

        // In the general case, the patterns in `pats` are
        // alternatives, so we must treat this like an N-way select
        // statement.
        let initial_state = reslice(in_out).to_vec();
        for pats.each |&pat| {
            let mut temp = copy initial_state;
            self.walk_pat(pat, temp, loop_scopes);
            join_bits(&self.dfcx.oper, temp, in_out);
        }
    }

    fn find_scope<'a>(&self,
                      expr: @ast::expr,
                      label: Option<ast::ident>,
                      loop_scopes: &'a mut ~[LoopScope]) -> &'a mut LoopScope {
        let index = match label {
            None => {
                let len = loop_scopes.len();
                len - 1
            }

            Some(_) => {
                match self.tcx().def_map.find(&expr.id) {
                    Some(&ast::def_label(loop_id)) => {
                        match loop_scopes.position(|l| l.loop_id == loop_id) {
                            Some(i) => i,
                            None => {
                                self.tcx().sess.span_bug(
                                    expr.span,
                                    fmt!("No loop scope for id %?", loop_id));
                            }
                        }
                    }

                    r => {
                        self.tcx().sess.span_bug(
                            expr.span,
                            fmt!("Bad entry `%?` in def_map for label", r));
                    }
                }
            }
        };

        &mut loop_scopes[index]
    }

    fn is_method_call(&self, expr: @ast::expr) -> bool {
        self.dfcx.method_map.contains_key(&expr.id)
    }

    fn reset(&mut self, bits: &mut [uint]) {
        let e = if self.dfcx.oper.initial_value() {uint::max_value} else {0};
        for vec::each_mut(bits) |b| { *b = e; }
    }

    fn add_to_entry_set(&mut self, id: ast::node_id, pred_bits: &[uint]) {
        debug!("add_to_entry_set(id=%?, pred_bits=%s)",
               id, bits_to_str(pred_bits));
        let (start, end) = self.dfcx.compute_id_range(id);
        let changed = { // FIXME(#5074) awkward construction
            let on_entry = vec::mut_slice(self.dfcx.on_entry, start, end);
            join_bits(&self.dfcx.oper, pred_bits, on_entry)
        };
        if changed {
            debug!("changed entry set for %? to %s",
                   id, bits_to_str(self.dfcx.on_entry.slice(start, end)));
            self.changed = true;
        }
    }

    fn merge_with_entry_set(&mut self,
                            id: ast::node_id,
                            pred_bits: &mut [uint]) {
        debug!("merge_with_entry_set(id=%?, pred_bits=%s)",
               id, mut_bits_to_str(pred_bits));
        let (start, end) = self.dfcx.compute_id_range(id);
        let changed = { // FIXME(#5074) awkward construction
            let on_entry = vec::mut_slice(self.dfcx.on_entry, start, end);
            let changed = join_bits(&self.dfcx.oper, reslice(pred_bits), on_entry);
            copy_bits(reslice(on_entry), pred_bits);
            changed
        };
        if changed {
            debug!("changed entry set for %? to %s",
                   id, bits_to_str(self.dfcx.on_entry.slice(start, end)));
            self.changed = true;
        }
    }
}

fn mut_bits_to_str(words: &mut [uint]) -> ~str {
    bits_to_str(reslice(words))
}

fn bits_to_str(words: &[uint]) -> ~str {
    let mut result = ~"";
    let mut sep = '[';

    // Note: this is a little endian printout of bytes.

    for words.each |&word| {
        let mut v = word;
        for uint::range(0, uint::bytes) |_| {
            str::push_char(&mut result, sep);
            str::push_str(&mut result, fmt!("%02x", v & 0xFF));
            v >>= 8;
            sep = '-';
        }
    }
    str::push_char(&mut result, ']');
    return result;
}

fn copy_bits(in_vec: &[uint], out_vec: &mut [uint]) -> bool {
    bitwise(out_vec, in_vec, |_, b| b)
}

fn join_bits<O:DataFlowOperator>(oper: &O,
                                 in_vec: &[uint],
                                 out_vec: &mut [uint]) -> bool {
    bitwise(out_vec, in_vec, |a, b| oper.join(a, b))
}

#[inline(always)]
fn bitwise(out_vec: &mut [uint],
           in_vec: &[uint],
           op: &fn(uint, uint) -> uint) -> bool {
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for uint::range(0, out_vec.len()) |i| {
        let old_val = out_vec[i];
        let new_val = op(old_val, in_vec[i]);
        out_vec[i] = new_val;
        changed |= (old_val != new_val);
    }
    return changed;
}

fn set_bit(words: &mut [uint], bit: uint) -> bool {
    debug!("set_bit: words=%s bit=%s",
           mut_bits_to_str(words), bit_str(bit));
    let word = bit / uint::bits;
    let bit_in_word = bit % uint::bits;
    let bit_mask = 1 << bit_in_word;
    debug!("word=%u bit_in_word=%u bit_mask=%u", word, bit_in_word, word);
    let oldv = words[word];
    let newv = oldv | bit_mask;
    words[word] = newv;
    oldv != newv
}

fn bit_str(bit: uint) -> ~str {
    let byte = bit >> 8;
    let lobits = 1 << (bit & 0xFF);
    fmt!("[%u:%u-%02x]", bit, byte, lobits)
}

fn reslice<'a>(v: &'a mut [uint]) -> &'a [uint] {
    // bFIXME(#5074) this function should not be necessary at all
    unsafe {
        cast::transmute(v)
    }
}

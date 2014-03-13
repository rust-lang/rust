// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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


use std::io;
use std::uint;
use std::vec;
use std::vec_ng::Vec;
use syntax::ast;
use syntax::ast_util;
use syntax::ast_util::IdRange;
use syntax::print::{pp, pprust};
use middle::ty;
use middle::typeck;
use util::ppaux::Repr;
use util::nodemap::NodeMap;

#[deriving(Clone)]
pub struct DataFlowContext<O> {
    priv tcx: ty::ctxt,
    priv method_map: typeck::MethodMap,

    /// the data flow operator
    priv oper: O,

    /// number of bits to propagate per id
    priv bits_per_id: uint,

    /// number of words we will use to store bits_per_id.
    /// equal to bits_per_id/uint::BITS rounded up.
    priv words_per_id: uint,

    // mapping from node to bitset index.
    priv nodeid_to_bitset: NodeMap<uint>,

    // Bit sets per id.  The following three fields (`gens`, `kills`,
    // and `on_entry`) all have the same structure. For each id in
    // `id_range`, there is a range of words equal to `words_per_id`.
    // So, to access the bits for any given id, you take a slice of
    // the full vector (see the method `compute_id_range()`).

    /// bits generated as we exit the scope `id`. Updated by `add_gen()`.
    priv gens: Vec<uint> ,

    /// bits killed as we exit the scope `id`. Updated by `add_kill()`.
    priv kills: Vec<uint> ,

    /// bits that are valid on entry to the scope `id`. Updated by
    /// `propagate()`.
    priv on_entry: Vec<uint> }

/// Parameterization for the precise form of data flow that is used.
pub trait DataFlowOperator {
    /// Specifies the initial value for each bit in the `on_entry` set
    fn initial_value(&self) -> bool;

    /// Joins two predecessor bits together, typically either `|` or `&`
    fn join(&self, succ: uint, pred: uint) -> uint;
}

struct PropagationContext<'a, O> {
    dfcx: &'a mut DataFlowContext<O>,
    changed: bool
}

struct LoopScope<'a> {
    loop_id: ast::NodeId,
    break_bits: Vec<uint>
}

impl<O:DataFlowOperator> pprust::PpAnn for DataFlowContext<O> {
    fn pre(&self, node: pprust::AnnNode) -> io::IoResult<()> {
        let (ps, id) = match node {
            pprust::NodeExpr(ps, expr) => (ps, expr.id),
            pprust::NodeBlock(ps, blk) => (ps, blk.id),
            pprust::NodeItem(ps, _) => (ps, 0),
            pprust::NodePat(ps, pat) => (ps, pat.id)
        };

        if self.nodeid_to_bitset.contains_key(&id) {
            let (start, end) = self.compute_id_range_frozen(id);
            let on_entry = self.on_entry.slice(start, end);
            let entry_str = bits_to_str(on_entry);

            let gens = self.gens.slice(start, end);
            let gens_str = if gens.iter().any(|&u| u != 0) {
                format!(" gen: {}", bits_to_str(gens))
            } else {
                ~""
            };

            let kills = self.kills.slice(start, end);
            let kills_str = if kills.iter().any(|&u| u != 0) {
                format!(" kill: {}", bits_to_str(kills))
            } else {
                ~""
            };

            let comment_str = format!("id {}: {}{}{}",
                                      id, entry_str, gens_str, kills_str);
            try!(pprust::synth_comment(ps, comment_str));
            try!(pp::space(&mut ps.s));
        }
        Ok(())
    }
}

impl<O:DataFlowOperator> DataFlowContext<O> {
    pub fn new(tcx: ty::ctxt,
               method_map: typeck::MethodMap,
               oper: O,
               id_range: IdRange,
               bits_per_id: uint) -> DataFlowContext<O> {
        let words_per_id = (bits_per_id + uint::BITS - 1) / uint::BITS;

        debug!("DataFlowContext::new(id_range={:?}, bits_per_id={:?}, words_per_id={:?})",
               id_range, bits_per_id, words_per_id);

        let gens = Vec::new();
        let kills = Vec::new();
        let on_entry = Vec::new();

        DataFlowContext {
            tcx: tcx,
            method_map: method_map,
            words_per_id: words_per_id,
            nodeid_to_bitset: NodeMap::new(),
            bits_per_id: bits_per_id,
            oper: oper,
            gens: gens,
            kills: kills,
            on_entry: on_entry
        }
    }

    pub fn add_gen(&mut self, id: ast::NodeId, bit: uint) {
        //! Indicates that `id` generates `bit`

        debug!("add_gen(id={:?}, bit={:?})", id, bit);
        let (start, end) = self.compute_id_range(id);
        {
            let gens = self.gens.mut_slice(start, end);
            set_bit(gens, bit);
        }
    }

    pub fn add_kill(&mut self, id: ast::NodeId, bit: uint) {
        //! Indicates that `id` kills `bit`

        debug!("add_kill(id={:?}, bit={:?})", id, bit);
        let (start, end) = self.compute_id_range(id);
        {
            let kills = self.kills.mut_slice(start, end);
            set_bit(kills, bit);
        }
    }

    fn apply_gen_kill(&mut self, id: ast::NodeId, bits: &mut [uint]) {
        //! Applies the gen and kill sets for `id` to `bits`

        debug!("apply_gen_kill(id={:?}, bits={}) [before]",
               id, mut_bits_to_str(bits));
        let (start, end) = self.compute_id_range(id);
        let gens = self.gens.slice(start, end);
        bitwise(bits, gens, |a, b| a | b);
        let kills = self.kills.slice(start, end);
        bitwise(bits, kills, |a, b| a & !b);

        debug!("apply_gen_kill(id={:?}, bits={}) [after]",
               id, mut_bits_to_str(bits));
    }

    fn apply_kill(&mut self, id: ast::NodeId, bits: &mut [uint]) {
        debug!("apply_kill(id={:?}, bits={}) [before]",
               id, mut_bits_to_str(bits));
        let (start, end) = self.compute_id_range(id);
        let kills = self.kills.slice(start, end);
        bitwise(bits, kills, |a, b| a & !b);
        debug!("apply_kill(id={:?}, bits={}) [after]",
               id, mut_bits_to_str(bits));
    }

    fn compute_id_range_frozen(&self, id: ast::NodeId) -> (uint, uint) {
        let n = *self.nodeid_to_bitset.get(&id);
        let start = n * self.words_per_id;
        let end = start + self.words_per_id;
        (start, end)
    }

    fn compute_id_range(&mut self, id: ast::NodeId) -> (uint, uint) {
        let mut expanded = false;
        let len = self.nodeid_to_bitset.len();
        let n = self.nodeid_to_bitset.find_or_insert_with(id, |_| {
            expanded = true;
            len
        });
        if expanded {
            let entry = if self.oper.initial_value() { uint::MAX } else {0};
            for _ in range(0, self.words_per_id) {
                self.gens.push(0);
                self.kills.push(0);
                self.on_entry.push(entry);
            }
        }
        let start = *n * self.words_per_id;
        let end = start + self.words_per_id;

        assert!(start < self.gens.len());
        assert!(end <= self.gens.len());
        assert!(self.gens.len() == self.kills.len());
        assert!(self.gens.len() == self.on_entry.len());

        (start, end)
    }


    pub fn each_bit_on_entry_frozen(&self,
                                    id: ast::NodeId,
                                    f: |uint| -> bool)
                                    -> bool {
        //! Iterates through each bit that is set on entry to `id`.
        //! Only useful after `propagate()` has been called.
        if !self.nodeid_to_bitset.contains_key(&id) {
            return true;
        }
        let (start, end) = self.compute_id_range_frozen(id);
        let on_entry = self.on_entry.slice(start, end);
        debug!("each_bit_on_entry_frozen(id={:?}, on_entry={})",
               id, bits_to_str(on_entry));
        self.each_bit(on_entry, f)
    }

    pub fn each_bit_on_entry(&mut self,
                             id: ast::NodeId,
                             f: |uint| -> bool)
                             -> bool {
        //! Iterates through each bit that is set on entry to `id`.
        //! Only useful after `propagate()` has been called.

        let (start, end) = self.compute_id_range(id);
        let on_entry = self.on_entry.slice(start, end);
        debug!("each_bit_on_entry(id={:?}, on_entry={})",
               id, bits_to_str(on_entry));
        self.each_bit(on_entry, f)
    }

    pub fn each_gen_bit(&mut self, id: ast::NodeId, f: |uint| -> bool)
                        -> bool {
        //! Iterates through each bit in the gen set for `id`.

        let (start, end) = self.compute_id_range(id);
        let gens = self.gens.slice(start, end);
        debug!("each_gen_bit(id={:?}, gens={})",
               id, bits_to_str(gens));
        self.each_bit(gens, f)
    }

    pub fn each_gen_bit_frozen(&self, id: ast::NodeId, f: |uint| -> bool)
                               -> bool {
        //! Iterates through each bit in the gen set for `id`.
        if !self.nodeid_to_bitset.contains_key(&id) {
            return true;
        }
        let (start, end) = self.compute_id_range_frozen(id);
        let gens = self.gens.slice(start, end);
        debug!("each_gen_bit(id={:?}, gens={})",
               id, bits_to_str(gens));
        self.each_bit(gens, f)
    }

    fn each_bit(&self, words: &[uint], f: |uint| -> bool) -> bool {
        //! Helper for iterating over the bits in a bit set.

        for (word_index, &word) in words.iter().enumerate() {
            if word != 0 {
                let base_index = word_index * uint::BITS;
                for offset in range(0u, uint::BITS) {
                    let bit = 1 << offset;
                    if (word & bit) != 0 {
                        // NB: we round up the total number of bits
                        // that we store in any given bit set so that
                        // it is an even multiple of uint::BITS.  This
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

impl<O:DataFlowOperator+Clone+'static> DataFlowContext<O> {
//                      ^^^^^^^^^^^^^ only needed for pretty printing
    pub fn propagate(&mut self, blk: &ast::Block) {
        //! Performs the data flow analysis.

        if self.bits_per_id == 0 {
            // Optimize the surprisingly common degenerate case.
            return;
        }

        {
            let mut propcx = PropagationContext {
                dfcx: self,
                changed: true
            };

            let mut temp = vec::from_elem(self.words_per_id, 0u);
            let mut loop_scopes = Vec::new();

            while propcx.changed {
                propcx.changed = false;
                propcx.reset(temp);
                propcx.walk_block(blk, temp, &mut loop_scopes);
            }
        }

        debug!("Dataflow result:");
        debug!("{}", {
            self.pretty_print_to(~io::stderr(), blk).unwrap();
            ""
        });
    }

    fn pretty_print_to(&self, wr: ~io::Writer,
                       blk: &ast::Block) -> io::IoResult<()> {
        let mut ps = pprust::rust_printer_annotated(wr, self);
        try!(pprust::cbox(&mut ps, pprust::indent_unit));
        try!(pprust::ibox(&mut ps, 0u));
        try!(pprust::print_block(&mut ps, blk));
        try!(pp::eof(&mut ps.s));
        Ok(())
    }
}

impl<'a, O:DataFlowOperator> PropagationContext<'a, O> {
    fn tcx(&self) -> ty::ctxt {
        self.dfcx.tcx
    }

    fn walk_block(&mut self,
                  blk: &ast::Block,
                  in_out: &mut [uint],
                  loop_scopes: &mut Vec<LoopScope> ) {
        debug!("DataFlowContext::walk_block(blk.id={}, in_out={})",
               blk.id, bits_to_str(in_out));

        self.merge_with_entry_set(blk.id, in_out);

        for &stmt in blk.stmts.iter() {
            self.walk_stmt(stmt, in_out, loop_scopes);
        }

        self.walk_opt_expr(blk.expr, in_out, loop_scopes);

        self.dfcx.apply_gen_kill(blk.id, in_out);
    }

    fn walk_stmt(&mut self,
                 stmt: @ast::Stmt,
                 in_out: &mut [uint],
                 loop_scopes: &mut Vec<LoopScope> ) {
        match stmt.node {
            ast::StmtDecl(decl, _) => {
                self.walk_decl(decl, in_out, loop_scopes);
            }

            ast::StmtExpr(expr, _) | ast::StmtSemi(expr, _) => {
                self.walk_expr(expr, in_out, loop_scopes);
            }

            ast::StmtMac(..) => {
                self.tcx().sess.span_bug(stmt.span, "unexpanded macro");
            }
        }
    }

    fn walk_decl(&mut self,
                 decl: @ast::Decl,
                 in_out: &mut [uint],
                 loop_scopes: &mut Vec<LoopScope> ) {
        match decl.node {
            ast::DeclLocal(local) => {
                self.walk_opt_expr(local.init, in_out, loop_scopes);
                self.walk_pat(local.pat, in_out, loop_scopes);
            }

            ast::DeclItem(_) => {}
        }
    }

    fn walk_expr(&mut self,
                 expr: &ast::Expr,
                 in_out: &mut [uint],
                 loop_scopes: &mut Vec<LoopScope> ) {
        debug!("DataFlowContext::walk_expr(expr={}, in_out={})",
               expr.repr(self.dfcx.tcx), bits_to_str(in_out));

        self.merge_with_entry_set(expr.id, in_out);

        match expr.node {
            ast::ExprFnBlock(..) | ast::ExprProc(..) => {
            }

            ast::ExprIf(cond, then, els) => {
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

                let mut then_bits = in_out.to_owned();
                self.walk_block(then, then_bits, loop_scopes);

                self.walk_opt_expr(els, in_out, loop_scopes);
                join_bits(&self.dfcx.oper, then_bits, in_out);
            }

            ast::ExprWhile(cond, blk) => {
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

                let mut body_bits = in_out.to_owned();
                loop_scopes.push(LoopScope {
                    loop_id: expr.id,
                    break_bits: Vec::from_slice(in_out)
                });
                self.walk_block(blk, body_bits, loop_scopes);
                self.add_to_entry_set(expr.id, body_bits);
                let new_loop_scope = loop_scopes.pop().unwrap();
                copy_bits(new_loop_scope.break_bits.as_slice(), in_out);
            }

            ast::ExprForLoop(..) => fail!("non-desugared expr_for_loop"),

            ast::ExprLoop(blk, _) => {
                //
                //     (expr) <--+
                //       |       |
                //       v       |
                //     (blk) ----+
                //       |
                //    <--+ (break)
                //

                let mut body_bits = in_out.to_owned();
                self.reset(in_out);
                loop_scopes.push(LoopScope {
                    loop_id: expr.id,
                    break_bits: Vec::from_slice(in_out)
                });
                self.walk_block(blk, body_bits, loop_scopes);
                self.add_to_entry_set(expr.id, body_bits);

                let new_loop_scope = loop_scopes.pop().unwrap();
                assert_eq!(new_loop_scope.loop_id, expr.id);
                copy_bits(new_loop_scope.break_bits.as_slice(), in_out);
            }

            ast::ExprMatch(discr, ref arms) => {
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

                let mut guards = in_out.to_owned();

                // We know that exactly one arm will be taken, so we
                // can start out with a blank slate and just union
                // together the bits from each arm:
                self.reset(in_out);

                for arm in arms.iter() {
                    // in_out reflects the discr and all guards to date
                    self.walk_opt_expr(arm.guard, guards, loop_scopes);

                    // determine the bits for the body and then union
                    // them into `in_out`, which reflects all bodies to date
                    let mut body = guards.to_owned();
                    self.walk_pat_alternatives(arm.pats.as_slice(),
                                               body,
                                               loop_scopes);
                    self.walk_expr(arm.body, body, loop_scopes);
                    join_bits(&self.dfcx.oper, body, in_out);
                }
            }

            ast::ExprRet(o_e) => {
                self.walk_opt_expr(o_e, in_out, loop_scopes);
                self.reset(in_out);
            }

            ast::ExprBreak(label) => {
                let scope = self.find_scope(expr, label, loop_scopes);
                self.break_from_to(expr, scope, in_out);
                self.reset(in_out);
            }

            ast::ExprAgain(label) => {
                let scope = self.find_scope(expr, label, loop_scopes);
                self.pop_scopes(expr, scope, in_out);
                self.add_to_entry_set(scope.loop_id, in_out);
                self.reset(in_out);
            }

            ast::ExprAssign(l, r) |
            ast::ExprAssignOp(_, l, r) => {
                self.walk_expr(r, in_out, loop_scopes);
                self.walk_expr(l, in_out, loop_scopes);
            }

            ast::ExprVec(ref exprs, _) => {
                self.walk_exprs(exprs.as_slice(), in_out, loop_scopes)
            }

            ast::ExprRepeat(l, r, _) => {
                self.walk_expr(l, in_out, loop_scopes);
                self.walk_expr(r, in_out, loop_scopes);
            }

            ast::ExprStruct(_, ref fields, with_expr) => {
                for field in fields.iter() {
                    self.walk_expr(field.expr, in_out, loop_scopes);
                }
                self.walk_opt_expr(with_expr, in_out, loop_scopes);
            }

            ast::ExprCall(f, ref args) => {
                self.walk_expr(f, in_out, loop_scopes);
                self.walk_call(expr.id, args.as_slice(), in_out, loop_scopes);
            }

            ast::ExprMethodCall(_, _, ref args) => {
                self.walk_call(expr.id, args.as_slice(), in_out, loop_scopes);
            }

            ast::ExprIndex(l, r) |
            ast::ExprBinary(_, l, r) if self.is_method_call(expr) => {
                self.walk_call(expr.id, [l, r], in_out, loop_scopes);
            }

            ast::ExprUnary(_, e) if self.is_method_call(expr) => {
                self.walk_call(expr.id, [e], in_out, loop_scopes);
            }

            ast::ExprTup(ref exprs) => {
                self.walk_exprs(exprs.as_slice(), in_out, loop_scopes);
            }

            ast::ExprBinary(op, l, r) if ast_util::lazy_binop(op) => {
                self.walk_expr(l, in_out, loop_scopes);
                let temp = in_out.to_owned();
                self.walk_expr(r, in_out, loop_scopes);
                join_bits(&self.dfcx.oper, temp, in_out);
            }

            ast::ExprIndex(l, r) |
            ast::ExprBinary(_, l, r) => {
                self.walk_exprs([l, r], in_out, loop_scopes);
            }

            ast::ExprLogLevel |
            ast::ExprLit(..) |
            ast::ExprPath(..) => {}

            ast::ExprAddrOf(_, e) |
            ast::ExprCast(e, _) |
            ast::ExprUnary(_, e) |
            ast::ExprParen(e) |
            ast::ExprVstore(e, _) |
            ast::ExprField(e, _, _) => {
                self.walk_expr(e, in_out, loop_scopes);
            }

            ast::ExprBox(s, e) => {
                self.walk_expr(s, in_out, loop_scopes);
                self.walk_expr(e, in_out, loop_scopes);
            }

            ast::ExprInlineAsm(ref inline_asm) => {
                for &(_, expr) in inline_asm.inputs.iter() {
                    self.walk_expr(expr, in_out, loop_scopes);
                }
                for &(_, expr) in inline_asm.outputs.iter() {
                    self.walk_expr(expr, in_out, loop_scopes);
                }
            }

            ast::ExprBlock(blk) => {
                self.walk_block(blk, in_out, loop_scopes);
            }

            ast::ExprMac(..) => {
                self.tcx().sess.span_bug(expr.span, "unexpanded macro");
            }
        }

        self.dfcx.apply_gen_kill(expr.id, in_out);
    }

    fn pop_scopes(&mut self,
                  from_expr: &ast::Expr,
                  to_scope: &mut LoopScope,
                  in_out: &mut [uint]) {
        //! Whenever you have a `break` or a `loop` statement, flow
        //! exits through any number of enclosing scopes on its
        //! way to the new destination. This function applies the kill
        //! sets of those enclosing scopes to `in_out` (those kill sets
        //! concern items that are going out of scope).

        let tcx = self.tcx();

        debug!("pop_scopes(from_expr={}, to_scope={:?}, in_out={})",
               from_expr.repr(tcx), to_scope.loop_id,
               bits_to_str(in_out));

        let mut id = from_expr.id;
        while id != to_scope.loop_id {
            self.dfcx.apply_kill(id, in_out);

            match tcx.region_maps.opt_encl_scope(id) {
                Some(i) => { id = i; }
                None => {
                    tcx.sess.span_bug(
                        from_expr.span,
                        format!("pop_scopes(from_expr={}, to_scope={:?}) \
                              to_scope does not enclose from_expr",
                             from_expr.repr(tcx), to_scope.loop_id));
                }
            }
        }
    }

    fn break_from_to(&mut self,
                     from_expr: &ast::Expr,
                     to_scope: &mut LoopScope,
                     in_out: &mut [uint]) {
        self.pop_scopes(from_expr, to_scope, in_out);
        self.dfcx.apply_kill(from_expr.id, in_out);
        join_bits(&self.dfcx.oper,
                  in_out,
                  to_scope.break_bits.as_mut_slice());
        debug!("break_from_to(from_expr={}, to_scope={}) final break_bits={}",
               from_expr.repr(self.tcx()),
               to_scope.loop_id,
               bits_to_str(in_out));
    }

    fn walk_exprs(&mut self,
                  exprs: &[@ast::Expr],
                  in_out: &mut [uint],
                  loop_scopes: &mut Vec<LoopScope> ) {
        for &expr in exprs.iter() {
            self.walk_expr(expr, in_out, loop_scopes);
        }
    }

    fn walk_opt_expr(&mut self,
                     opt_expr: Option<@ast::Expr>,
                     in_out: &mut [uint],
                     loop_scopes: &mut Vec<LoopScope> ) {
        for &expr in opt_expr.iter() {
            self.walk_expr(expr, in_out, loop_scopes);
        }
    }

    fn walk_call(&mut self,
                 call_id: ast::NodeId,
                 args: &[@ast::Expr],
                 in_out: &mut [uint],
                 loop_scopes: &mut Vec<LoopScope> ) {
        self.walk_exprs(args, in_out, loop_scopes);

        // FIXME(#6268) nested method calls
        // self.merge_with_entry_set(in_out);
        // self.dfcx.apply_gen_kill(in_out);

        let return_ty = ty::node_id_to_type(self.tcx(), call_id);
        let fails = ty::type_is_bot(return_ty);
        if fails {
            self.reset(in_out);
        }
    }

    fn walk_pat(&mut self,
                pat: @ast::Pat,
                in_out: &mut [uint],
                _loop_scopes: &mut Vec<LoopScope> ) {
        debug!("DataFlowContext::walk_pat(pat={}, in_out={})",
               pat.repr(self.dfcx.tcx), bits_to_str(in_out));

        ast_util::walk_pat(pat, |p| {
            debug!("  p.id={} in_out={}", p.id, bits_to_str(in_out));
            self.merge_with_entry_set(p.id, in_out);
            self.dfcx.apply_gen_kill(p.id, in_out);
            true
        });
    }

    fn walk_pat_alternatives(&mut self,
                             pats: &[@ast::Pat],
                             in_out: &mut [uint],
                             loop_scopes: &mut Vec<LoopScope> ) {
        if pats.len() == 1 {
            // Common special case:
            return self.walk_pat(pats[0], in_out, loop_scopes);
        }

        // In the general case, the patterns in `pats` are
        // alternatives, so we must treat this like an N-way select
        // statement.
        let initial_state = in_out.to_owned();
        for &pat in pats.iter() {
            let mut temp = initial_state.clone();
            self.walk_pat(pat, temp, loop_scopes);
            join_bits(&self.dfcx.oper, temp, in_out);
        }
    }

    fn find_scope<'a,'b>(
                  &self,
                  expr: &ast::Expr,
                  label: Option<ast::Ident>,
                  loop_scopes: &'a mut Vec<LoopScope<'b>>)
                  -> &'a mut LoopScope<'b> {
        let index = match label {
            None => {
                let len = loop_scopes.len();
                len - 1
            }

            Some(_) => {
                let def_map = self.tcx().def_map.borrow();
                match def_map.get().find(&expr.id) {
                    Some(&ast::DefLabel(loop_id)) => {
                        match loop_scopes.iter().position(|l| l.loop_id == loop_id) {
                            Some(i) => i,
                            None => {
                                self.tcx().sess.span_bug(
                                    expr.span,
                                    format!("no loop scope for id {:?}", loop_id));
                            }
                        }
                    }

                    r => {
                        self.tcx().sess.span_bug(
                            expr.span,
                            format!("bad entry `{:?}` in def_map for label", r));
                    }
                }
            }
        };

        loop_scopes.get_mut(index)
    }

    fn is_method_call(&self, expr: &ast::Expr) -> bool {
        let method_call = typeck::MethodCall::expr(expr.id);
        self.dfcx.method_map.borrow().get().contains_key(&method_call)
    }

    fn reset(&mut self, bits: &mut [uint]) {
        let e = if self.dfcx.oper.initial_value() {uint::MAX} else {0};
        for b in bits.mut_iter() { *b = e; }
    }

    fn add_to_entry_set(&mut self, id: ast::NodeId, pred_bits: &[uint]) {
        debug!("add_to_entry_set(id={:?}, pred_bits={})",
               id, bits_to_str(pred_bits));
        let (start, end) = self.dfcx.compute_id_range(id);
        let changed = { // FIXME(#5074) awkward construction
            let on_entry = self.dfcx.on_entry.mut_slice(start, end);
            join_bits(&self.dfcx.oper, pred_bits, on_entry)
        };
        if changed {
            debug!("changed entry set for {:?} to {}",
                   id, bits_to_str(self.dfcx.on_entry.slice(start, end)));
            self.changed = true;
        }
    }

    fn merge_with_entry_set(&mut self,
                            id: ast::NodeId,
                            pred_bits: &mut [uint]) {
        debug!("merge_with_entry_set(id={:?}, pred_bits={})",
               id, mut_bits_to_str(pred_bits));
        let (start, end) = self.dfcx.compute_id_range(id);
        let changed = { // FIXME(#5074) awkward construction
            let on_entry = self.dfcx.on_entry.mut_slice(start, end);
            let changed = join_bits(&self.dfcx.oper, pred_bits, on_entry);
            copy_bits(on_entry, pred_bits);
            changed
        };
        if changed {
            debug!("changed entry set for {:?} to {}",
                   id, bits_to_str(self.dfcx.on_entry.slice(start, end)));
            self.changed = true;
        }
    }
}

fn mut_bits_to_str(words: &mut [uint]) -> ~str {
    bits_to_str(words)
}

fn bits_to_str(words: &[uint]) -> ~str {
    let mut result = ~"";
    let mut sep = '[';

    // Note: this is a little endian printout of bytes.

    for &word in words.iter() {
        let mut v = word;
        for _ in range(0u, uint::BYTES) {
            result.push_char(sep);
            result.push_str(format!("{:02x}", v & 0xFF));
            v >>= 8;
            sep = '-';
        }
    }
    result.push_char(']');
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

#[inline]
fn bitwise(out_vec: &mut [uint], in_vec: &[uint], op: |uint, uint| -> uint)
           -> bool {
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for (out_elt, in_elt) in out_vec.mut_iter().zip(in_vec.iter()) {
        let old_val = *out_elt;
        let new_val = op(old_val, *in_elt);
        *out_elt = new_val;
        changed |= old_val != new_val;
    }
    changed
}

fn set_bit(words: &mut [uint], bit: uint) -> bool {
    debug!("set_bit: words={} bit={}",
           mut_bits_to_str(words), bit_str(bit));
    let word = bit / uint::BITS;
    let bit_in_word = bit % uint::BITS;
    let bit_mask = 1 << bit_in_word;
    debug!("word={} bit_in_word={} bit_mask={}", word, bit_in_word, word);
    let oldv = words[word];
    let newv = oldv | bit_mask;
    words[word] = newv;
    oldv != newv
}

fn bit_str(bit: uint) -> ~str {
    let byte = bit >> 8;
    let lobits = 1 << (bit & 0xFF);
    format!("[{}:{}-{:02x}]", bit, byte, lobits)
}

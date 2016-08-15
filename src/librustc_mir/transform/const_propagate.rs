// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is Constant-Simplify propagation pass. This is a composition of two distinct
//! passes: constant-propagation and terminator simplification.
//!
//! Note that having these passes interleaved results in a strictly more potent optimisation pass
//! which is able to optimise CFG in a way which two passes couldn’t do separately even if they
//! were run indefinitely one after another.
//!
//! Both these passes are very similar in their nature:
//!
//!                  | Constant  | Simplify  |
//! |----------------|-----------|-----------|
//! | Lattice Domain | Lvalue    | Lvalue    |
//! | Lattice Value  | Constant  | Constant  |
//! | Transfer       | x = const | x = const |
//! | Rewrite        | x → const | T(x) → T' |
//! | Top            | {}        | {}        |
//! | Join           | intersect | intersect |
//! |----------------|-----------|-----------|
//!
//! It might be a good idea to eventually interleave move propagation here, as it has very
//! similar lattice to the constant propagation pass.

use rustc_data_structures::fnv::FnvHashMap;
use rustc::middle::const_val::ConstVal;
use rustc::hir::def_id::DefId;
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::mir::visit::{MutVisitor};
use rustc::ty::TyCtxt;
use rustc_const_eval::{eval_const_binop, eval_const_unop, cast_const};

use super::dataflow::*;

pub struct ConstPropagate;

impl Pass for ConstPropagate {}

impl<'tcx> MirPass<'tcx> for ConstPropagate {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut Mir<'tcx>) {
        *mir = Dataflow::forward(mir, ConstLattice::new(),
                                 ConstTransfer { tcx: tcx },
                                 ConstRewrite { tcx: tcx }.and_then(SimplifyRewrite));
    }
}

#[derive(Debug, Clone)]
struct ConstLattice<'tcx> {
    statics: FnvHashMap<DefId, Constant<'tcx>>,
    // key must not be a static or a projection.
    locals: FnvHashMap<Lvalue<'tcx>, Constant<'tcx>>,
}

impl<'tcx> ConstLattice<'tcx> {
    fn new() -> ConstLattice<'tcx> {
        ConstLattice {
            statics: FnvHashMap(),
            locals: FnvHashMap()
        }
    }

    fn insert(&mut self, key: &Lvalue<'tcx>, val: Constant<'tcx>) {
        // FIXME: HashMap has no way to insert stuff without cloning the key even if it exists
        // already.
        match *key {
            Lvalue::Static(defid) => self.statics.insert(defid, val),
            // I feel like this could be handled, but needs special care. For example in code like
            // this:
            //
            // ```
            // var.field = false;
            // something(&mut var);
            // assert!(var.field);
            // ```
            //
            // taking a reference to var should invalidate knowledge about all the
            // projections of var and not just var itself. Currently we handle this by not
            // keeping any knowledge about projections at all, but I think eventually we
            // want to do so.
            Lvalue::Projection(_) => None,
            _ => self.locals.insert(key.clone(), val)
        };
    }

    fn get(&self, key: &Lvalue<'tcx>) -> Option<&Constant<'tcx>> {
        match *key {
            Lvalue::Static(ref defid) => self.statics.get(defid),
            Lvalue::Projection(_) => None,
            _ => self.locals.get(key),
        }
    }

    fn top(&mut self, key: &Lvalue<'tcx>) {
        match *key {
            Lvalue::Static(ref defid) => { self.statics.remove(defid); }
            Lvalue::Projection(_) => {}
            _ => { self.locals.remove(key); }
        }
    }
}

fn intersect_map<K, V>(this: &mut FnvHashMap<K, V>, mut other: FnvHashMap<K, V>) -> bool
where K: Eq + ::std::hash::Hash, V: PartialEq
{
    let mut changed = false;
    // Calculate inteersection this way:
    // First, drain the self.values into a list of equal values common to both maps.
    let mut common_keys = vec![];
    for (key, value) in this.drain() {
        match other.remove(&key) {
            None => {
                // self had the key, but not other, so it is Top (i.e. removed)
                changed = true;
            }
            Some(ov) => {
                // Similarly, if the values are not equal…
                changed |= if ov != value {
                    true
                } else {
                    common_keys.push((key, value));
                    false
                }
            }
        }
    }
    // Now, put each common key with equal value back into the map.
    for (key, value) in common_keys {
        this.insert(key, value);
    }
    changed
}

impl<'tcx> Lattice for ConstLattice<'tcx> {
    fn bottom() -> Self { unimplemented!() }
    fn join(&mut self, other: Self) -> bool {
        intersect_map(&mut self.locals, other.locals) |
        intersect_map(&mut self.statics, other.statics)
    }
}

struct ConstTransfer<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> Transfer<'tcx> for ConstTransfer<'a, 'tcx> {
    type Lattice = ConstLattice<'tcx>;
    type TerminatorReturn = Vec<Self::Lattice>;

    fn stmt(&self, s: &Statement<'tcx>, mut lat: Self::Lattice)
    -> Self::Lattice
    {
        let (lval, rval) = match s.kind {
            // Could be handled with some extra information, but there’s no point in doing that yet
            // because currently we would have no consumers of this data (no GetDiscriminant).
            StatementKind::SetDiscriminant { .. } => return lat,
            StatementKind::StorageDead(ref lval) => {
                lat.top(lval);
                return lat;
            }
            StatementKind::StorageLive(_) => return lat,
            StatementKind::Assign(ref lval, ref rval) => (lval, rval),
        };
        match *rval {
            Rvalue::Use(Operand::Consume(_)) => {
                lat.top(lval);
            },
            Rvalue::Use(Operand::Constant(ref cnst)) => {
                lat.insert(lval, cnst.clone());
            },
            // We do not want to deal with references and pointers here. Not yet and not without
            // a way to query stuff about reference/pointer aliasing.
            Rvalue::Ref(_, _, ref referee) => {
                lat.top(lval);
                lat.top(referee);
            }
            // FIXME: should calculate length of statically sized arrays and store it.
            Rvalue::Len(_) => { lat.top(lval); }
            // FIXME: should keep length around for Len case above.
            Rvalue::Repeat(_, _) => { lat.top(lval); }
            // Not handled. Constant casts should turn out as plain ConstVals by here.
            Rvalue::Cast(_, _, _) => { lat.top(lval); }
            // Make sure case like `var1 = var1 {op} x` does not make our knowledge incorrect.
            Rvalue::BinaryOp(..) | Rvalue::CheckedBinaryOp(..) | Rvalue::UnaryOp(..) => {
                lat.top(lval);
            }
            // Cannot be handled
            Rvalue::Box(_) => { lat.top(lval); }
            // Not handled, but could be. Disaggregation helps to not bother with this.
            Rvalue::Aggregate(..) => { lat.top(lval); }
            // Not handled, invalidate any knowledge about any variables touched by this. Dangerous
            // stuff and other dragons be here.
            Rvalue::InlineAsm { ref outputs, ref inputs, asm: _ } => {
                lat.top(lval);
                for output in outputs { lat.top(output); }
                for input in inputs {
                    if let Operand::Consume(ref lval) = *input { lat.top(lval); }
                }
                // Clear the statics, because inline assembly may mutate global state at will.
                lat.statics.clear();
            }
        };
        lat
    }

    fn term(&self, t: &Terminator<'tcx>, mut lat: Self::Lattice)
    -> Self::TerminatorReturn
    {
        let span = t.source_info.span;
        let succ_count = t.successors().len();
        let bool_const = |b: bool| Constant {
            span: span,
            ty: self.tcx.mk_bool(),
            literal: Literal::Value { value: ConstVal::Bool(b) },
        };
        match t.kind {
            TerminatorKind::If { cond: Operand::Consume(ref lval), .. } => {
                let mut falsy = lat.clone();
                falsy.insert(lval, bool_const(false));
                lat.insert(lval, bool_const(true));
                vec![lat, falsy]
            }
            TerminatorKind::SwitchInt { ref discr, ref values, switch_ty, .. } => {
                let mut vec: Vec<_> = values.iter().map(|val| {
                    let mut branch = lat.clone();
                    branch.insert(discr, Constant {
                        span: span,
                        ty: switch_ty,
                        literal: Literal::Value { value: val.clone() }
                    });
                    branch
                }).collect();
                vec.push(lat);
                vec
            }
            TerminatorKind::Drop { ref location, .. } => {
                lat.top(location);
                // See comment in Call.
                lat.statics.clear();
                vec![lat; succ_count]
            }
            TerminatorKind::DropAndReplace { ref location, ref value, .. } => {
                match *value {
                    Operand::Consume(_) => lat.top(location),
                    Operand::Constant(ref cnst) => lat.insert(location, cnst.clone()),
                }
                // See comment in Call.
                lat.statics.clear();
                vec![lat; succ_count]
            }
            TerminatorKind::Call { ref destination, ref args, .. } => {
                for arg in args {
                    if let Operand::Consume(ref lval) = *arg {
                        // FIXME: Probably safe to not remove any non-projection lvals.
                        lat.top(lval);
                    }
                }
                destination.as_ref().map(|&(ref lval, _)| lat.top(lval));
                // Clear all knowledge about statics, because call may mutate any global state
                // without us knowing about it.
                lat.statics.clear();
                vec![lat; succ_count]
            }
            TerminatorKind::Assert { ref cond, expected, ref cleanup, .. } => {
                if let Operand::Consume(ref lval) = *cond {
                    lat.insert(lval, bool_const(expected));
                    if cleanup.is_some() {
                        let mut falsy = lat.clone();
                        falsy.insert(lval, bool_const(!expected));
                        vec![lat, falsy]
                    } else {
                        vec![lat]
                    }
                } else {
                    vec![lat; succ_count]
                }
            }
            TerminatorKind::Switch { .. } | // Might make some sense to handle this
            TerminatorKind::If { .. } | // The condition is constant
                                        // (unreachable if interleaved with simplify branches pass)
            TerminatorKind::Goto { .. } |
            TerminatorKind::Unreachable |
            TerminatorKind::Return |
            TerminatorKind::Resume => {
                vec![lat; succ_count]
            }
        }
    }
}

struct ConstRewrite<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}
impl<'a, 'tcx, T> Rewrite<'tcx, T> for ConstRewrite<'a, 'tcx>
where T: Transfer<'tcx, Lattice=ConstLattice<'tcx>>
{
    fn stmt(&self, stmt: &Statement<'tcx>, fact: &T::Lattice) -> StatementChange<'tcx> {
        let mut stmt = stmt.clone();
        let mut vis = RewriteConstVisitor(&fact);
        vis.visit_statement(START_BLOCK, &mut stmt);
        ConstEvalVisitor(self.tcx).visit_statement(START_BLOCK, &mut stmt);
        StatementChange::Statement(stmt)
    }

    fn term(&self, term: &Terminator<'tcx>, fact: &T::Lattice) -> TerminatorChange<'tcx> {
        let mut term = term.clone();
        let mut vis = RewriteConstVisitor(&fact);
        vis.visit_terminator(START_BLOCK, &mut term);
        ConstEvalVisitor(self.tcx).visit_terminator(START_BLOCK, &mut term);
        TerminatorChange::Terminator(term)
    }
}

struct RewriteConstVisitor<'a, 'tcx: 'a>(&'a ConstLattice<'tcx>);
impl<'a, 'tcx> MutVisitor<'tcx> for RewriteConstVisitor<'a, 'tcx> {
    fn visit_operand(&mut self, op: &mut Operand<'tcx>) {
        // To satisy borrow checker, modify `op` after inspecting it
        let repl = if let Operand::Consume(ref lval) = *op {
            self.0.get(lval).cloned()
        } else {
            None
        };
        if let Some(c) = repl {
            *op = Operand::Constant(c);
        }
        self.super_operand(op);
    }
}
struct ConstEvalVisitor<'a, 'tcx: 'a>(TyCtxt<'a, 'tcx, 'tcx>);
impl<'a, 'tcx> MutVisitor<'tcx> for ConstEvalVisitor<'a, 'tcx> {
    fn visit_statement(&mut self, _: BasicBlock, stmt: &mut Statement<'tcx>) {
        let span = stmt.source_info.span;
        let rval = if let StatementKind::Assign(_, ref mut rval) = stmt.kind {
            rval
        } else {
            return
        };
        let repl = match *rval {
            // FIXME: Rvalue::CheckedBinaryOp could be evaluated to Rvalue::Aggregate of 2-tuple
            // (or disaggregated version of it; needs replacement with arbitrary graphs)
            Rvalue::BinaryOp(ref op, Operand::Constant(ref opr1), Operand::Constant(ref opr2)) => {
                match (&opr1.literal, &opr2.literal) {
                    (&Literal::Value { value: ref value1 },
                     &Literal::Value { value: ref value2 }) =>
                        eval_const_binop(op.to_hir_binop(), &value1, &value2, span).ok().map(|v| {
                            (v, op.ty(self.0, opr1.ty, opr2.ty))
                        }),
                    _ => None
                }
            }
            Rvalue::UnaryOp(ref op, Operand::Constant(ref opr1)) => {
                if let Literal::Value { ref value } = opr1.literal {
                    eval_const_unop(op.to_hir_unop(), value, span).ok().map(|v| (v, opr1.ty))
                } else {
                    None
                }
            }
            Rvalue::Cast(CastKind::Misc, Operand::Constant(ref opr), ty) => {
                let ret = if let Literal::Value { ref value } = opr.literal {
                    cast_const(self.0, value.clone(), ty).ok().map(|v| (v, ty))
                } else {
                    None
                };
                ret
            }
            _ => None
        };
        if let Some((constant, ty)) = repl {
            *rval = Rvalue::Use(Operand::Constant(Constant {
                span: span,
                ty: ty,
                literal: Literal::Value { value: constant }
            }));
        }

        self.super_rvalue(rval)
    }
}

struct SimplifyRewrite;

impl<'tcx, T: Transfer<'tcx>> Rewrite<'tcx, T> for SimplifyRewrite {
    fn stmt(&self, s: &Statement<'tcx>, _: &T::Lattice)
    -> StatementChange<'tcx> {
        StatementChange::Statement(s.clone())
    }

    fn term(&self, t: &Terminator<'tcx>, _: &T::Lattice)
    -> TerminatorChange<'tcx> {
        match t.kind {
            TerminatorKind::If { ref targets, .. } if targets.0 == targets.1 => {
                let mut nt = t.clone();
                nt.kind = TerminatorKind::Goto { target: targets.0 };
                TerminatorChange::Terminator(nt)
            }
            TerminatorKind::If { ref targets, cond: Operand::Constant(Constant {
                literal: Literal::Value {
                    value: ConstVal::Bool(cond)
                }, ..
            }) } => {
                let mut nt = t.clone();
                if cond {
                    nt.kind = TerminatorKind::Goto { target: targets.0 };
                } else {
                    nt.kind = TerminatorKind::Goto { target: targets.1 };
                }
                TerminatorChange::Terminator(nt)
            }
            TerminatorKind::SwitchInt { ref targets, .. } if targets.len() == 1 => {
                let mut nt = t.clone();
                nt.kind = TerminatorKind::Goto { target: targets[0] };
                TerminatorChange::Terminator(nt)
            }
            TerminatorKind::Assert { target, cond: Operand::Constant(Constant {
                literal: Literal::Value {
                    value: ConstVal::Bool(cond)
                }, ..
            }), expected, .. } if cond == expected => {
                // FIXME: once replacements with arbitrary subgraphs get implemented, this should
                // have success branch pointed to a block with Unreachable terminator when cond !=
                // expected.
                let mut nt = t.clone();
                nt.kind = TerminatorKind::Goto { target: target };
                TerminatorChange::Terminator(nt)
            }
            _ => TerminatorChange::Terminator(t.clone())
        }
    }
}

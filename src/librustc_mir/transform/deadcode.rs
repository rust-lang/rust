// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::mir::visit::{Visitor, LvalueContext};
use rustc::ty::TyCtxt;

use super::dataflow::*;

pub struct DeadCode;

impl Pass for DeadCode {}

impl<'tcx> MirPass<'tcx> for DeadCode {
    fn run_pass<'a>(&mut self, _: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut Mir<'tcx>) {
        let new_mir = Dataflow::backward(mir, DeadCodeLattice::bottom(),
                                         DeadCodeTransfer, DeadCodeRewrite);
        *mir = new_mir;
    }
}

#[derive(Debug, Clone)]
struct DeadCodeLattice {
    vars: BitVector,
    args: BitVector,
    tmps: BitVector,
}

impl Lattice for DeadCodeLattice {
    fn bottom() -> Self {
        DeadCodeLattice {
            vars: BitVector::new(0),
            tmps: BitVector::new(0),
            args: BitVector::new(0)
        }
    }
    fn join(&mut self, other: Self) -> bool {
        self.vars.grow(other.vars.len());
        self.tmps.grow(other.tmps.len());
        self.args.grow(other.args.len());
        let (r1, r2, r3) = (self.vars.insert_all(&other.vars)
                           ,self.tmps.insert_all(&other.tmps)
                           ,self.args.insert_all(&other.args));
        r1 || r2 || r3
    }
}

impl DeadCodeLattice {
    fn set_lvalue_live<'a>(&mut self, l: &Lvalue<'a>) {
        match *l {
            Lvalue::Arg(a) => {
                self.args.grow(a.index() + 1);
                self.args.insert(a.index());
            }
            Lvalue::Temp(t) => {
                self.tmps.grow(t.index() + 1);
                self.tmps.insert(t.index());
            }
            Lvalue::Var(v) => {
                self.vars.grow(v.index() + 1);
                self.vars.insert(v.index());
            }
            _ => {}
        }
    }

    fn set_lvalue_dead<'a>(&mut self, l: &Lvalue<'a>) {
        match *l.base() {
            Lvalue::Arg(a) => self.args.remove(a.index()),
            Lvalue::Temp(t) => self.tmps.remove(t.index()),
            Lvalue::Var(v) => self.vars.remove(v.index()),
            _ => false
        };
    }
}

struct DeadCodeTransfer;
impl<'tcx> Transfer<'tcx> for DeadCodeTransfer {
    type Lattice = DeadCodeLattice;
    type TerminatorReturn = DeadCodeLattice;

    fn stmt(&self, s: &Statement<'tcx>, lat: DeadCodeLattice) -> DeadCodeLattice {
        let mut vis = DeadCodeVisitor(lat);
        vis.visit_statement(START_BLOCK, s);
        vis.0
    }

    fn term(&self, t: &Terminator<'tcx>, lat: DeadCodeLattice) -> DeadCodeLattice {
        let mut vis = DeadCodeVisitor(lat);
        vis.visit_terminator(START_BLOCK, t);
        vis.0
    }
}

struct DeadCodeRewrite;
impl<'tcx, T> Rewrite<'tcx, T> for DeadCodeRewrite
where T: Transfer<'tcx, Lattice=DeadCodeLattice>
{
    fn stmt(&self, s: &Statement<'tcx>, lat: &DeadCodeLattice)
    -> StatementChange<'tcx>
    {
        let StatementKind::Assign(ref lval, ref rval) = s.kind;
        let keep = !rval.is_pure() || match *lval {
            Lvalue::Temp(t) => lat.tmps.contains(t.index()),
            Lvalue::Var(v) => lat.vars.contains(v.index()),
            Lvalue::Arg(a) => lat.args.contains(a.index()),
            _ => true
        };
        if keep {
            StatementChange::Statement(s.clone())
        } else {
            StatementChange::Remove
        }
    }

    fn term(&self, t: &Terminator<'tcx>, _: &DeadCodeLattice)
    -> TerminatorChange<'tcx>
    {
        TerminatorChange::Terminator(t.clone())
    }
}

struct DeadCodeVisitor(DeadCodeLattice);
impl<'tcx> Visitor<'tcx> for DeadCodeVisitor {
    fn visit_lvalue(&mut self, lval: &Lvalue<'tcx>, ctx: LvalueContext) {
        if ctx == LvalueContext::Store || ctx == LvalueContext::CallStore {
            match *lval {
                // This is a assign to the variable in a way that all uses dominated by this store
                // do not count as live.
                ref l@Lvalue::Temp(_) |
                ref l@Lvalue::Var(_) |
                ref l@Lvalue::Arg(_) => self.0.set_lvalue_dead(l),
                _ => {}
            }
        } else {
            self.0.set_lvalue_live(lval);
        }
        self.super_lvalue(lval, ctx);
    }
}

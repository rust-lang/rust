// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is Alias-Constant-Simplify propagation pass. This is a composition of three distinct
//! dataflow passes: alias-propagation, constant-propagation and terminator simplification.
//!
//! All these are very similar in their nature:
//!
//!                 | Constant  |  Alias   | Simplify  |
//!|----------------|-----------|----------|-----------|
//!| Lattice Domain | Lvalue    | Lvalue   | Lvalue    |
//!| Lattice Value  | Constant  | Lvalue   | Constant  |
//!| Transfer       | x = const | x = lval | x = const |
//!| Rewrite        | x → const | x → lval | T(x) → T' |
//!| Bottom         | {}        | {}       | {}        |
//!
//! For all of them we will be using a lattice of Hashmap from Lvalue to
//! WTop<Either<Lvalue, Constant>>
//!
//! My personal belief is that it should be possible to make a way to compose two hashmap lattices
//! into one, but I can’t seem to get it just right yet, so we do the composing and decomposing
//! manually here.

use rustc_data_structures::fnv::FnvHashMap;
use rustc::mir::repr::*;
use rustc::mir::visit::{MutVisitor, LvalueContext};
use rustc::mir::transform::lattice::{Lattice, WBottom};
use rustc::mir::transform::dataflow::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::ty::TyCtxt;
use rustc::middle::const_val::ConstVal;
use pretty;

#[derive(PartialEq, Debug, Eq, Clone)]
enum Either<'tcx> {
    Lvalue(Lvalue<'tcx>),
    Const(Constant<'tcx>),
}

#[derive(Debug, Clone)]
struct AcsLattice<'tcx> {
    known_values: FnvHashMap<Lvalue<'tcx>, Either<'tcx>>
}

impl<'tcx> Lattice for AcsLattice<'tcx> {
    fn bottom() -> Self { unimplemented!() }
    fn join(&mut self, other: &Self) -> bool {
        let mut to_remove = vec![];

        for (k, v) in &self.known_values {
            if other.known_values.get(k).map_or(true, |other_v| other_v != v) {
                to_remove.push(k.clone());
            }
        }

        for k in &to_remove {
            self.known_values.remove(k);
        }

        !to_remove.is_empty()
    }
}

pub struct AcsPropagate;

impl Pass for AcsPropagate {}

impl<'tcx> MirPass<'tcx> for AcsPropagate {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource, mir: &mut Mir<'tcx>) {
        let ret = ar_forward(
            &mut mir.cfg,
            Facts::new(),
            AcsPropagateTransfer,
            AliasRewrite.and_then(ConstRewrite).and_then(SimplifyRewrite)
        );
        mir.cfg = ret.0;
        pretty::dump_mir(tcx, "acs_propagate", &0, src, mir, None);
    }

}

struct AcsPropagateTransfer;

fn base_lvalue<'a, 'tcx>(mut lval: &'a Lvalue<'tcx>) -> &'a Lvalue<'tcx> {
    while let &Lvalue::Projection(ref proj) = lval {
        lval = &proj.base;
    }
    lval
}

fn invalidate<'tcx>(map: &mut FnvHashMap<Lvalue<'tcx>, Either<'tcx>>, lval: &Lvalue<'tcx>) {
    map.remove(lval);

    let mut repl = None;

    for (k, v) in &mut *map {
        if let Either::Lvalue(ref mut nlval) = *v {
            if nlval == lval {
                match repl {
                    None => {
                        repl = Some(k.clone())
                    },
                    Some(ref r) => {
                        *nlval = r.clone();
                    }
                }
            }
        }
    }

    if let Some(repl) = repl {
        map.remove(&repl);
    }
}

impl<'tcx> Transfer<'tcx> for AcsPropagateTransfer {
    type Lattice = WBottom<AcsLattice<'tcx>>;

    fn stmt(&self, s: &Statement<'tcx>, lat: WBottom<AcsLattice<'tcx>>) -> WBottom<AcsLattice<'tcx>> {
        let mut lat_map = match lat {
            WBottom::Bottom => FnvHashMap::default(),
            WBottom::Value(lat) => lat.known_values
        };

        let StatementKind::Assign(ref lval, ref rval) = s.kind;
        invalidate(&mut lat_map, base_lvalue(lval));

        if let &Lvalue::Projection(_) = lval {
            return WBottom::Value(AcsLattice {
                known_values: lat_map
            });
        }

        match *rval {
            Rvalue::Use(Operand::Consume(ref nlval)) => {
                lat_map.insert(lval.clone(), Either::Lvalue(nlval.clone()));
            },
            Rvalue::Use(Operand::Constant(ref c)) => {
                lat_map.insert(lval.clone(), Either::Const(c.clone()));
            },
            _ => { }
        };

        WBottom::Value(AcsLattice {
            known_values: lat_map
        })
    }

    fn term(&self, t: &Terminator<'tcx>, mut lat: WBottom<AcsLattice<'tcx>>) -> Vec<WBottom<AcsLattice<'tcx>>> {
        match t.kind {
            TerminatorKind::Call { .. } |
            TerminatorKind::Drop { .. } |
            TerminatorKind::DropAndReplace { .. } => {
                // FIXME: Be smarter here by using an alias analysis
                lat = WBottom::Value(AcsLattice {
                    known_values: FnvHashMap::default()
                });
            },
            _ => { }
        }

        // FIXME: this should inspect the terminators and set their known values to constants. Esp.
        // for the if: in the truthy branch the operand is known to be true and in the falsy branch
        // the operand is known to be false. Now we just ignore the potential here.
        let mut ret = vec![];
        ret.resize(t.successors().len(), lat);
        ret
    }
}

struct AliasRewrite;

impl<'tcx> Rewrite<'tcx, WBottom<AcsLattice<'tcx>>> for AliasRewrite {
    fn stmt(&self, s: &Statement<'tcx>, l: &WBottom<AcsLattice<'tcx>>, _: &mut CFG<'tcx>)
    -> StatementChange<'tcx> {
        if let &WBottom::Value(ref lat) = l {
            let mut ns = s.clone();
            let mut vis = RewriteAliasVisitor(&lat.known_values, false);
            vis.visit_statement(START_BLOCK, &mut ns);
            if vis.1 {
                return StatementChange::Statement(ns);
            }
        }
        StatementChange::None
    }

    fn term(&self, t: &Terminator<'tcx>, l: &WBottom<AcsLattice<'tcx>>, _: &mut CFG<'tcx>)
    -> TerminatorChange<'tcx> {
        if let &WBottom::Value(ref lat) = l {
            let mut nt = t.clone();
            let mut vis = RewriteAliasVisitor(&lat.known_values, false);
            vis.visit_terminator(START_BLOCK, &mut nt);
            if vis.1 {
                return TerminatorChange::Terminator(nt);
            }
        }
        TerminatorChange::None
    }
}

struct RewriteAliasVisitor<'a, 'tcx: 'a>(&'a FnvHashMap<Lvalue<'tcx>, Either<'tcx>>, bool);
impl<'a, 'tcx> MutVisitor<'tcx> for RewriteAliasVisitor<'a, 'tcx> {
    fn visit_lvalue(&mut self, lvalue: &mut Lvalue<'tcx>, context: LvalueContext) {
        match context {
            LvalueContext::Consume => {
                if let Some(&Either::Lvalue(ref nlval)) = self.0.get(lvalue) {
                    self.1 = true;
                    *lvalue = nlval.clone();
                }
            },
            _ => { }
        }
        self.super_lvalue(lvalue, context);
    }
}

struct ConstRewrite;

impl<'tcx> Rewrite<'tcx, WBottom<AcsLattice<'tcx>>> for ConstRewrite {
    fn stmt(&self, s: &Statement<'tcx>, l: &WBottom<AcsLattice<'tcx>>, _: &mut CFG<'tcx>)
    -> StatementChange<'tcx> {
        if let &WBottom::Value(ref lat) = l {
            let mut ns = s.clone();
            let mut vis = RewriteConstVisitor(&lat.known_values, false);
            vis.visit_statement(START_BLOCK, &mut ns);
            if vis.1 {
                return StatementChange::Statement(ns);
            }
        }
        StatementChange::None
    }

    fn term(&self, t: &Terminator<'tcx>, l: &WBottom<AcsLattice<'tcx>>, _: &mut CFG<'tcx>)
    -> TerminatorChange<'tcx> {
        if let &WBottom::Value(ref lat) = l {
            let mut nt = t.clone();
            let mut vis = RewriteConstVisitor(&lat.known_values, false);
            vis.visit_terminator(START_BLOCK, &mut nt);
            if vis.1 {
                return TerminatorChange::Terminator(nt);
            }
        }
        TerminatorChange::None
    }
}

struct RewriteConstVisitor<'a, 'tcx: 'a>(&'a FnvHashMap<Lvalue<'tcx>, Either<'tcx>>, bool);
impl<'a, 'tcx> MutVisitor<'tcx> for RewriteConstVisitor<'a, 'tcx> {
    fn visit_operand(&mut self, op: &mut Operand<'tcx>) {
        // To satisy borrow checker, modify `op` after inspecting it
        let repl = if let Operand::Consume(ref lval) = *op {
            if let Some(&Either::Const(ref c)) = self.0.get(lval) {
                Some(c.clone())
            } else {
                None
            }
        } else {
            None
        };
        if let Some(c) = repl {
            self.1 = true;
            *op = Operand::Constant(c);
        }

        self.super_operand(op);
    }
}


struct SimplifyRewrite;

impl<'tcx, L: Lattice> Rewrite<'tcx, L> for SimplifyRewrite {
    fn stmt(&self, _: &Statement<'tcx>, _: &L, _: &mut CFG<'tcx>)
    -> StatementChange<'tcx> {
        StatementChange::None
    }

    fn term(&self, t: &Terminator<'tcx>, _: &L, _: &mut CFG<'tcx>)
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
            _ => TerminatorChange::None
        }
    }
}

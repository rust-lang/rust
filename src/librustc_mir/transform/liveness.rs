use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::mir::visit::{Visitor, LvalueContext};
use rustc::ty::TyCtxt;

use super::dataflow::*;

pub struct LocalLivenessAnalysis;

impl Pass for LocalLivenessAnalysis {}

impl<'tcx> MirPass<'tcx> for LocalLivenessAnalysis {
    fn run_pass<'a>(&mut self, _: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut Mir<'tcx>) {
        let new_mir = Dataflow::backward(mir, LiveValueTransfer, LiveValueRewrite);
        *mir = new_mir;
    }
}

#[derive(Debug, Clone)]
struct LiveValueLattice {
    vars: BitVector,
    args: BitVector,
    tmps: BitVector,
}

impl Lattice for LiveValueLattice {
    fn bottom() -> Self {
        LiveValueLattice {
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

impl LiveValueLattice {
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

struct LiveValueTransfer;
impl<'tcx> Transfer<'tcx> for LiveValueTransfer {
    type Lattice = LiveValueLattice;
    type TerminatorReturn = LiveValueLattice;

    fn stmt(&self, s: &Statement<'tcx>, lat: LiveValueLattice) -> LiveValueLattice {
        let mut vis = LiveValueVisitor(lat);
        vis.visit_statement(START_BLOCK, s);
        vis.0
    }

    fn term(&self, t: &Terminator<'tcx>, lat: LiveValueLattice) -> LiveValueLattice {
        let mut vis = LiveValueVisitor(lat);
        vis.visit_terminator(START_BLOCK, t);
        vis.0
    }
}

struct LiveValueRewrite;
impl<'tcx, T> Rewrite<'tcx, T> for LiveValueRewrite
where T: Transfer<'tcx, Lattice=LiveValueLattice>
{
    fn stmt(&self, s: &Statement<'tcx>, lat: &LiveValueLattice)
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

    fn term(&self, t: &Terminator<'tcx>, _: &LiveValueLattice)
    -> TerminatorChange<'tcx>
    {
        TerminatorChange::Terminator(t.clone())
    }
}

struct LiveValueVisitor(LiveValueLattice);
impl<'tcx> Visitor<'tcx> for LiveValueVisitor {
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

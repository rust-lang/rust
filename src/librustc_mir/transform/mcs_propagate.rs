use rustc_data_structures::fnv::FnvHashMap;
use rustc::middle::const_val::ConstVal;
use rustc::mir::repr::*;
use rustc::mir::tcx::binop_ty;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::mir::visit::{MutVisitor, LvalueContext};
use rustc::ty::TyCtxt;
use std::collections::hash_map::Entry;
use rustc_const_eval::{eval_const_binop, eval_const_unop, cast_const};

use super::dataflow::*;

pub struct McsPropagate;

impl Pass for McsPropagate {
    fn name(&self) -> &'static str { "McsPropagate" }
}

impl<'tcx> MirPass<'tcx> for McsPropagate {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut Mir<'tcx>) {
        *mir = Dataflow::forward(mir, McsTransfer { tcx: tcx },
                                 // MoveRewrite.and_then(ConstRewrite).and_then(SimplifyRewrite)
                                 ConstRewrite { tcx: tcx }.and_then(SimplifyRewrite));
    }
}

#[derive(PartialEq, Debug, Clone)]
enum Either<'tcx> {
    Lvalue(Lvalue<'tcx>),
    Const(Constant<'tcx>),
}

#[derive(Debug, Clone)]
struct McsLattice<'tcx> {
    values: FnvHashMap<Lvalue<'tcx>, Either<'tcx>>
}

impl<'tcx> Lattice for McsLattice<'tcx> {
    fn bottom() -> Self { McsLattice { values: FnvHashMap() } }
    fn join(&mut self, mut other: Self) -> bool {
        // Calculate inteersection this way:
        let mut changed = false;

        // First, drain the self.values into a list of equal values common to both maps.
        let mut common_keys = vec![];
        for (key, value) in self.values.drain() {
            match other.values.remove(&key) {
                    // self had the key, but not other, so removing
                None => changed = true,
                Some(ov) => if ov.eq(&value) {
                    // common key, equal value
                    common_keys.push((key, value))
                } else {
                    // both had key, but different values, so removing
                    changed = true;
                },
            }
        }
        // Now, put each common key with equal value back into the map.
        for (key, value) in common_keys {
            self.values.insert(key, value);
        }
        changed
    }
}

struct McsTransfer<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx> Transfer<'tcx> for McsTransfer<'a, 'tcx> {
    type Lattice = McsLattice<'tcx>;
    type TerminatorReturn = Vec<Self::Lattice>;

    fn stmt(&self, s: &Statement<'tcx>, lat: Self::Lattice)
    -> Self::Lattice
    {
        let StatementKind::Assign(ref lval, ref rval) = s.kind;
        match *lval {
            // Do not bother with statics – global state.
            Lvalue::Static(_) => return lat,
            // I feel like this could be handled, but needs special care. For example in code like
            // this:
            //
            // ```
            // var.field = false;
            // something(&mut var);
            // assert!(var.field);
            // ```
            //
            // taking a reference to var should invalidate knowledge about all the projections of
            // var and not just var itself. Currently we handle this by not keeping any knowledge
            // about projections at all, but I think eventually we want to do so.
            Lvalue::Projection(_) => return lat,
            _ => {}
        }
        let mut map = lat.values;
        match *rval {
            Rvalue::Use(Operand::Consume(ref nlval)) => {
                map.insert(lval.clone(), Either::Lvalue(nlval.clone()));
            },
            Rvalue::Use(Operand::Constant(ref cnst)) => {
                map.insert(lval.clone(), Either::Const(cnst.clone()));
            },
            // We do not want to deal with references and pointers here. Not yet and not without
            // a way to query stuff about reference/pointer aliasing.
            Rvalue::Ref(_, _, ref referee) => {
                map.remove(lval);
                map.remove(referee);
            }
            // TODO: should calculate length of statically sized arrays
            Rvalue::Len(_) => { map.remove(lval); }
            // TODO: should keep length around for Len case above.
            Rvalue::Repeat(_, _) => { map.remove(lval); }
            // Not handled. Constant casts should turn out as plain ConstVals by here.
            Rvalue::Cast(_, _, _) => { map.remove(lval); }
            // Make sure case like `var1 = var1 {op} x` does not make our knowledge incorrect.
            Rvalue::BinaryOp(..) | Rvalue::CheckedBinaryOp(..) | Rvalue::UnaryOp(..) => {
                map.remove(lval);
            }
            // Cannot be handled
            Rvalue::Box(_) => {  map.remove(lval); }
            // Not handled, but could be. Disaggregation helps to not bother with this.
            Rvalue::Aggregate(..) => { map.remove(lval); }
            // Not handled, invalidate any knowledge about any variables used by this. Dangerous
            // stuff and other dragons be here.
            Rvalue::InlineAsm { ref outputs, ref inputs, asm: _ } => {
                map.remove(lval);
                for output in outputs { map.remove(output); }
                for input in inputs {
                    if let Operand::Consume(ref lval) = *input { map.remove(lval); }
                }
            }
        };
        McsLattice { values: map }
    }

    fn term(&self, t: &Terminator<'tcx>, lat: Self::Lattice)
    -> Self::TerminatorReturn
    {
        let mut map = lat.values;
        let span = t.source_info.span;
        let succ_count = t.successors().len();
        let bool_const = |b: bool| Either::Const(Constant {
            span: span,
            ty: self.tcx.mk_bool(),
            literal: Literal::Value { value: ConstVal::Bool(b) },
        });
        let wrap = |v| McsLattice { values: v };
        match t.kind {
            TerminatorKind::If { cond: Operand::Consume(ref lval), .. } => {
                let mut falsy = map.clone();
                falsy.insert(lval.clone(), bool_const(false));
                map.insert(lval.clone(), bool_const(true));
                vec![wrap(map), wrap(falsy)]
            }
            TerminatorKind::SwitchInt { ref discr, ref values, switch_ty, .. } => {
                let mut vec: Vec<_> = values.iter().map(|val| {
                    let mut map = map.clone();
                    map.insert(discr.clone(), Either::Const(Constant {
                        span: span,
                        ty: switch_ty,
                        literal: Literal::Value { value: val.clone() }
                    }));
                    wrap(map)
                }).collect();
                vec.push(wrap(map));
                vec
            }
            TerminatorKind::Drop { ref location, ref unwind, .. } => {
                let mut map = map.clone();
                map.remove(location);
                if unwind.is_some() {
                    vec![wrap(map.clone()), wrap(map)]
                } else {
                    vec![wrap(map)]
                }
            }
            TerminatorKind::DropAndReplace { ref location, ref unwind, ref value, .. } => {
                let value = match *value {
                    Operand::Consume(ref lval) => Either::Lvalue(lval.clone()),
                    Operand::Constant(ref cnst) => Either::Const(cnst.clone()),
                };
                map.insert(location.clone(), value);
                if unwind.is_some() {
                    let mut unwind = map.clone();
                    unwind.remove(location);
                    vec![wrap(map), wrap(unwind)]
                } else {
                    vec![wrap(map)]
                }
            }
            TerminatorKind::Call { ref destination, ref args, .. } => {
                for arg in args {
                    if let Operand::Consume(ref lval) = *arg {
                        // TODO(nagisa): Probably safe to not remove any non-projection lvals.
                        map.remove(lval);
                    }
                }
                destination.as_ref().map(|&(ref lval, _)| map.remove(lval));
                vec![wrap(map); succ_count]
            }
            TerminatorKind::Assert { ref cond, expected, ref cleanup, .. } => {
                if let Operand::Consume(ref lval) = *cond {
                    map.insert(lval.clone(), bool_const(expected));
                    if cleanup.is_some() {
                        let mut falsy = map.clone();
                        falsy.insert(lval.clone(), bool_const(!expected));
                        vec![wrap(map), wrap(falsy)]
                    } else {
                        vec![wrap(map)]
                    }
                } else {
                    vec![wrap(map); succ_count]
                }
            }
            TerminatorKind::Switch { .. } | // Might make some sense to handle this
            TerminatorKind::If { .. } | // The condition is constant
            TerminatorKind::Goto { .. } |
            TerminatorKind::Unreachable |
            TerminatorKind::Return |
            TerminatorKind::Resume => {
                vec![wrap(map); succ_count]
            }
        }
    }
}

struct MoveRewrite;

impl<'tcx, T> Rewrite<'tcx, T> for MoveRewrite
where T: Transfer<'tcx, Lattice=McsLattice<'tcx>>
{
    fn stmt(&self, stmt: &Statement<'tcx>, fact: &T::Lattice) -> StatementChange<'tcx> {
        let mut stmt = stmt.clone();
        let mut vis = RewriteMoveVisitor(&fact.values);
        vis.visit_statement(START_BLOCK, &mut stmt);
        StatementChange::Statement(stmt)
    }

    fn term(&self, term: &Terminator<'tcx>, fact: &T::Lattice) -> TerminatorChange<'tcx> {
        let mut term = term.clone();
        let mut vis = RewriteMoveVisitor(&fact.values);
        vis.visit_terminator(START_BLOCK, &mut term);
        TerminatorChange::Terminator(term)
    }
}

struct RewriteMoveVisitor<'a, 'tcx: 'a>(&'a FnvHashMap<Lvalue<'tcx>, Either<'tcx>>);
impl<'a, 'tcx> MutVisitor<'tcx> for RewriteMoveVisitor<'a, 'tcx> {
    fn visit_lvalue(&mut self, lvalue: &mut Lvalue<'tcx>, context: LvalueContext) {
        match context {
            LvalueContext::Consume => {
                if let Some(&Either::Lvalue(ref nlval)) = self.0.get(lvalue) {
                    *lvalue = nlval.clone();
                }
            },
            _ => { }
        }
        self.super_lvalue(lvalue, context);
    }
}

struct ConstRewrite<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx, T> Rewrite<'tcx, T> for ConstRewrite<'a, 'tcx>
where T: Transfer<'tcx, Lattice=McsLattice<'tcx>>
{
    fn stmt(&self, stmt: &Statement<'tcx>, fact: &T::Lattice) -> StatementChange<'tcx> {
        let mut stmt = stmt.clone();
        let mut vis = RewriteConstVisitor(&fact.values);
        vis.visit_statement(START_BLOCK, &mut stmt);
        ConstEvalVisitor(self.tcx).visit_statement(START_BLOCK, &mut stmt);
        StatementChange::Statement(stmt)
    }

    fn term(&self, term: &Terminator<'tcx>, fact: &T::Lattice) -> TerminatorChange<'tcx> {
        let mut term = term.clone();
        let mut vis = RewriteConstVisitor(&fact.values);
        vis.visit_terminator(START_BLOCK, &mut term);
        ConstEvalVisitor(self.tcx).visit_terminator(START_BLOCK, &mut term);
        TerminatorChange::Terminator(term)
    }
}

struct RewriteConstVisitor<'a, 'tcx: 'a>(&'a FnvHashMap<Lvalue<'tcx>, Either<'tcx>>);
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
            *op = Operand::Constant(c);
        }
        self.super_operand(op);
    }
}
struct ConstEvalVisitor<'a, 'tcx: 'a>(TyCtxt<'a, 'tcx, 'tcx>);
impl<'a, 'tcx> MutVisitor<'tcx> for ConstEvalVisitor<'a, 'tcx> {
    fn visit_statement(&mut self, _: BasicBlock, stmt: &mut Statement<'tcx>) {
        let span = stmt.source_info.span;
        let StatementKind::Assign(_, ref mut rval) = stmt.kind;
        let repl = match *rval {
            // TODO: Rvalue::CheckedBinaryOp could be evaluated to Rvalue::Aggregate of 2-tuple (or
            // disaggregated version of it)
            Rvalue::BinaryOp(ref op, Operand::Constant(ref opr1), Operand::Constant(ref opr2)) => {
                match (&opr1.literal, &opr2.literal) {
                    (&Literal::Value { value: ref value1 },
                     &Literal::Value { value: ref value2 }) =>
                        eval_const_binop(op.to_hir_binop(), &value1, &value2, span).ok().map(|v| {
                            (v, binop_ty(self.0, *op, opr1.ty, opr2.ty))
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

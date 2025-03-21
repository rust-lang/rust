use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::*;
use rustc_session::Session;

use crate::MirPass;
/// This pass simplifes MIR by replacing places based on past Rvalues.
/// For example, this MIR:
/// ```text
/// _2 = Some(_1)
/// _3 = (_2 as Some).0
/// ```text
/// Will get simplfied into this MIR:
/// ```text
/// _2 = Some(_1)
/// _3 = _1
/// ```
/// This pass can also propagate uses of locals:
/// ```text
/// _2 = copy _1.0
/// _3 = copy _2.1
/// ```
/// ```text
/// _2 = copy _1.0
/// _3 = copy _1.0.1
/// ```
/// To simplify the implementation, this pass has some limitations:
/// 1. It will never propagate any rvalue across a write to memory.
/// 2. It never propagates rvalues with moves.
pub(super) struct PropRvalues;
impl<'tcx> MirPass<'tcx> for PropRvalues {
    /// This pass is relatively cheap, and(by itself) does not affect the debug information whatsoever.
    /// FIXME: check if this pass would be benficial to enable for mir_opt_level = 1
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.mir_opt_level() > 1
    }
    fn is_required(&self) -> bool {
        false
    }
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // This pass has a O(n^2) memory usage, so I limit it to <= 256 locals.
        // FIXME: check if this limit can be raised.
        if body.local_decls.len() > 256 {
            return;
        }

        let mut prop = PropagateLocals::new(body.local_decls.len(), tcx);
        // We don't care about the location here.
        let dummy = Location { block: BasicBlock::ZERO, statement_index: 0 };
        // We consider each block separately - this simplifes the implementation considerably.
        for block in body.basic_blocks.as_mut_preserves_cfg() {
            for statement in &mut block.statements {
                prop.visit_statement(statement, dummy);
            }
            // This allows us to *sometimes* elide needless copies.
            if let Some(ref mut terminator) = block.terminator {
                prop.visit_terminator(terminator, dummy);
            };
            prop.reset();
        }
    }
}
struct PropagateLocals<'tcx> {
    locals: IndexVec<Local, Option<Rvalue<'tcx>>>,
    /// Contains the list of rvalues which are invalidated if local `if_modifed` is modifed.
    /// \[if_modifed\]\[should_invalidate\]
    deps: IndexVec<Local, DenseBitSet<Local>>,
    tcx: TyCtxt<'tcx>,
}
impl<'tcx> PropagateLocals<'tcx> {
    /// Registers that `target_local` depends on `local`, and will be invalidated once `local` is written to.
    fn register_dep(&mut self, target_local: Local, local: Local) {
        self.deps[local].insert(target_local);
    }
    /// Marks `local` as potentially written to, invalidating all rvalues which reference it
    fn write_local(&mut self, local: Local) {
        for set in self.deps[local].iter() {
            self.locals[set] = None;
        }
    }
    fn new(locals: usize, tcx: TyCtxt<'tcx>) -> Self {
        Self {
            locals: IndexVec::from_elem_n(None, locals),
            deps: IndexVec::from_elem_n(DenseBitSet::new_empty(locals), locals),
            tcx,
        }
    }
    /// Resets the propagator, marking all rvalues as invalid, and clearing dependency info.
    fn reset(&mut self) {
        self.locals.iter_mut().for_each(|l| *l = None);
        self.deps.iter_mut().for_each(|si| si.clear());
    }
    /// Checks what rvalues are invalidated by `lvalue`, and removes them from the propagation process.
    fn invalidate_place(&mut self, lvalue: &Place<'_>) {
        // if this contains *deref* then the *lvalue* could be writing to anything.
        if lvalue.projection.contains(&ProjectionElem::Deref) {
            self.reset();
        }
        self.write_local(lvalue.local);
    }
    /// Adds rvalue to the propagation list, if eligible.
    /// Rvalue is eligible for propagation if:
    /// 1. It does not read from memory(no Deref projection)
    /// 2. It does not move from any locals(since that invalidates them).
    /// 3. lvalue it is assigned to is a local.
    /// This function also automaticaly invalidates all locals this rvalue is moving.
    fn add_rvalue(&mut self, rvalue: &Rvalue<'tcx>, lvalue: &Place<'tcx>) {
        struct HasDerfOrMoves<'b, 'tcx> {
            has_deref: bool,
            has_moves: bool,
            prop: &'b mut PropagateLocals<'tcx>,
            target_local: Local,
        }
        impl<'tcx, 'b> Visitor<'tcx> for HasDerfOrMoves<'b, 'tcx> {
            fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
                match operand {
                    Operand::Move(place) => {
                        self.has_moves = true;
                        // Exact semantics of moves are not decided yet, so I *assume* they leave behind unintialized memory.
                        self.prop.invalidate_place(place);
                    }
                    _ => (),
                }
                self.super_operand(operand, location);
            }
            fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
                // if this contains *deref* then the *rvalue* could be reading anything.
                if place.projection.contains(&ProjectionElem::Deref) {
                    self.has_deref = true;
                }
                self.super_place(place, ctx, loc);
            }
            fn visit_local(&mut self, local: Local, _: PlaceContext, _: Location) {
                self.prop.register_dep(self.target_local, local);
            }
        }
        // Check if this rvalue has derefs or moves, and invalidate all moved locals.
        let HasDerfOrMoves { has_deref, has_moves, .. } = {
            let mut vis = HasDerfOrMoves {
                has_deref: false,
                has_moves: false,
                prop: self,
                target_local: lvalue.local,
            };
            // We don't care about the location here.
            let dummy = Location { block: BasicBlock::ZERO, statement_index: 0 };
            vis.visit_rvalue(rvalue, dummy);
            vis
        };
        // Reads from memory, so moving it around is not always sound.
        if has_deref {
            self.write_local(lvalue.local);
            return;
        }
        // Has moves, can't be soundly duplicated / moved around (semantics of moves are undecided, so this may leave unitialized memory behind).
        if has_moves {
            self.write_local(lvalue.local);
            return;
        }
        // Add to the propagation list, if this rvalue is directly assigned to a local.
        if let Some(local) = lvalue.as_local() {
            self.locals[local] = Some(rvalue.clone());
        } else {
            self.write_local(lvalue.local);
        }
    }
}
impl<'tcx> MutVisitor<'tcx> for PropagateLocals<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    /// Visit an operand, propagating rvalues along the way.
    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, _: Location) {
        use rustc_middle::mir::Rvalue::*;
        // Constant - rvalues can't be propagated
        let Some(place) = operand.place() else {
            return;
        };
        // No rvalue registered for this local, so we can't propagate anything.
        let Some(ref rval) = self.locals[place.local] else {
            return;
        };
        // *For now*, this only handles aggregates and direct uses.
        // however, this can be easily extended in the future, to add support for more rvalues and places
        // (eg. for removing unneeded transmutes)
        match (&place.projection[..], rval) {
            ([], Use(Operand::Copy(src_place))) => *operand = Operand::Copy(src_place.clone()),
            (
                [ProjectionElem::Downcast(_, variant), PlaceElem::Field(field_idx, _)],
                Aggregate(box AggregateKind::Adt(_, var, _, _, active), fields),
            ) => {
                if variant == var && active.is_none() {
                    let Some(fplace) = fields[*field_idx].place().clone() else {
                        return;
                    };
                    *operand = Operand::Copy(fplace);
                }
            }
            _ => (),
        }
    }
    fn visit_assign(&mut self, lvalue: &mut Place<'tcx>, rvalue: &mut Rvalue<'tcx>, loc: Location) {
        // Propagating rvalues/places for Ref is not sound, since :
        // _2 = copy _1
        // _3 = &_2
        // is not equivalent to:
        // _2 = copy _1
        // _3 = &_1
        if matches!(rvalue, Rvalue::Ref(..) | Rvalue::RawPtr(..)) {
            return;
        }
        self.super_rvalue(rvalue, loc);
        self.invalidate_place(lvalue);
        self.add_rvalue(rvalue, lvalue);
    }
    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, loc: Location) {
        use rustc_middle::mir::StatementKind::*;
        match &mut statement.kind {
            Assign(_) | FakeRead(_) | PlaceMention(_) | AscribeUserType(..) => {
                self.super_statement(statement, loc)
            }
            // StorageDead and Deinit invalidates `loc`, cause they may deinitialize that local.
            StorageDead(loc) => self.write_local(*loc),
            Deinit(place) => self.write_local(place.local),
            // SetDiscriminant invalidates `loc`, since it could turn, eg. Ok(u32) to Err(u32).
            SetDiscriminant { place, .. } => self.write_local(place.local),
            // FIXME: *do retags invalidate the local*? Per docs, this "reads and modifies the place in an opaque way".
            // So, I assume this invalidates the local to be sure.
            Retag(_, place) => self.write_local(place.local),
            // FIXME: should coverage invalidate all locals?
            // I conservatively assume I can't propagate *any* locals across a coverage statement,
            // because that *could* cause a computation to be ascribed to wrong coverage info.
            Coverage(..) => self.locals.iter_mut().for_each(|l| *l = None),
            // FIXME: intrinsics *almost certainly* don't read / write to any locals whose address has not been taken,
            // but I am still unsure if moving a computation across them is safe. So, I don't do that for now.
            Intrinsic(_) => self.locals.iter_mut().for_each(|l| *l = None),
            // StorageLive and ConstEvalCounter can't invalidate a local.
            StorageLive(_) => (),
            // Nop and Nop-like statements.
            ConstEvalCounter | Nop | BackwardIncompatibleDropHint { .. } => (),
        }
    }
}

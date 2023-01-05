use crate::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{NonUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

const SUCCESSOR_LIMIT: usize = 100;

pub struct SinkConstAssignments;

impl<'tcx> MirPass<'tcx> for SinkConstAssignments {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 1
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // The primary benefit of this pass is sinking assignments to drop flags and enabling
        // ConstGoto and SimplifyCfg to merge the drop flag check into existing control flow.
        // If we permit sinking assignments to any local, we will sometimes sink an assignment into
        // but not completely through a goto chain, preventing SimplifyCfg from removing the
        // blocks.
        let mut optimizable_locals = branched_locals(body);
        if optimizable_locals.is_empty() {
            return;
        }

        let borrowed_locals = rustc_mir_dataflow::impls::borrowed_locals(body);
        optimizable_locals.subtract(&borrowed_locals);
        if optimizable_locals.is_empty() {
            return;
        }

        'outer: for block in 0..body.basic_blocks.len() {
            let block = block.into();
            let block_data = &body.basic_blocks[block];
            let Some(terminator) = &block_data.terminator else { continue; };

            let mut successors = Vec::new();
            for succ in terminator.successors() {
                // Successors which are just a Resume are okay
                if is_empty_resume(&body.basic_blocks[succ]) {
                    continue;
                }
                if body.basic_blocks.predecessors()[succ].len() != 1 {
                    debug!("Bailing from {block:?} because {succ:?} has multiple predecessors");
                    continue 'outer;
                }
                successors.push(succ);
            }

            if successors.len() > SUCCESSOR_LIMIT {
                debug!("Will not sink assignment, its basic block has too many successors");
            }

            let mut local_uses = None;
            for statement_idx in 0..body.basic_blocks[block].statements.len() {
                let statement = &body.basic_blocks[block].statements[statement_idx];
                if let StatementKind::Assign(box (place, Rvalue::Use(Operand::Constant(_)))) =
                    &statement.kind
                {
                    let local = place.local;
                    if !place.projection.is_empty() {
                        debug!("Nonempty place projection: {statement:?}");
                        continue;
                    }
                    if !optimizable_locals.contains(local) {
                        continue;
                    }

                    let uses = match &local_uses {
                        Some(uses) => uses,
                        None => {
                            let mut visitor = CountUsesVisitor::new();
                            visitor.visit_basic_block_data(block, &body.basic_blocks[block]);
                            local_uses = Some(visitor);
                            local_uses.as_ref().unwrap()
                        }
                    };

                    // If the local dies in this block, don't propagate it
                    if uses.dead.contains(&local) {
                        continue;
                    }
                    if !uses.is_used_once(local) {
                        debug!("Local used elsewhere in this block: {statement:?}");
                        continue;
                    }
                    if !tcx.consider_optimizing(|| format!("Sinking const assignment to {local:?}"))
                    {
                        debug!("optimization fuel exhausted");
                        break 'outer;
                    }
                    debug!("Sinking const assignment to {local:?}");
                    let blocks = body.basic_blocks.as_mut_preserves_cfg();
                    let statement = blocks[block].statements[statement_idx].replace_nop();

                    for succ in &successors {
                        let mut successor_uses = SuccessorUsesVisitor::new(local);
                        successor_uses.visit_basic_block_data(*succ, &blocks[*succ]);

                        if let Some(used) = successor_uses.first_use() {
                            if used == blocks[*succ].statements.len() {
                                blocks[*succ].statements.push(statement.clone());
                                continue;
                            }
                            // If the first use of our local in this block is another const
                            // assignment to it, do not paste a new assignment right before it
                            // because that would just create dead code.
                            if let StatementKind::Assign(box (
                                place,
                                Rvalue::Use(Operand::Constant(_)),
                            )) = &blocks[*succ].statements[used].kind
                            {
                                if place.local == local && place.projection.is_empty() {
                                    continue;
                                }
                            }
                            blocks[*succ].statements.insert(used, statement.clone());
                        } else {
                            blocks[*succ].statements.push(statement.clone());
                        }
                    }
                }
            }
        }
    }
}

fn branched_locals(body: &Body<'_>) -> BitSet<Local> {
    let mut visitor = BranchedLocals {
        branched: BitSet::new_empty(body.local_decls.len()),
        gets_const_assign: BitSet::new_empty(body.local_decls.len()),
    };
    visitor.visit_body(body);
    visitor.branched.intersect(&visitor.gets_const_assign);
    visitor.branched
}

struct BranchedLocals {
    branched: BitSet<Local>,
    gets_const_assign: BitSet<Local>,
}

impl Visitor<'_> for BranchedLocals {
    fn visit_terminator(&mut self, terminator: &Terminator<'_>, _location: Location) {
        let TerminatorKind::SwitchInt { discr, .. } = &terminator.kind else { return; };
        if let Some(place) = discr.place() {
            self.branched.insert(place.local);
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'_>, _location: Location) {
        if let StatementKind::Assign(box (place, Rvalue::Use(Operand::Constant(_)))) =
            &statement.kind
        {
            if place.projection.is_empty() {
                self.gets_const_assign.insert(place.local);
            }
        }
    }
}

fn is_empty_resume<'tcx>(block: &BasicBlockData<'tcx>) -> bool {
    block.statements.iter().all(|s| matches!(s.kind, StatementKind::Nop))
        && block.terminator.as_ref().map(|t| &t.kind) == Some(&TerminatorKind::Resume)
}

struct CountUsesVisitor {
    counts: FxHashMap<Local, usize>,
    dead: FxHashSet<Local>,
}

impl CountUsesVisitor {
    fn new() -> Self {
        Self { counts: FxHashMap::default(), dead: FxHashSet::default() }
    }

    fn is_used_once(&self, local: Local) -> bool {
        self.counts.get(&local) == Some(&1)
    }
}

impl Visitor<'_> for CountUsesVisitor {
    fn visit_local(&mut self, local: Local, context: PlaceContext, _location: Location) {
        match context {
            PlaceContext::NonUse(NonUseContext::StorageDead) => {
                self.dead.insert(local);
            }
            PlaceContext::NonUse(_) => {}
            PlaceContext::MutatingUse(_) | PlaceContext::NonMutatingUse(_) => {
                *self.counts.entry(local).or_default() += 1;
            }
        };
    }
}

struct SuccessorUsesVisitor {
    local: Local,
    first_use: Option<usize>,
}

impl SuccessorUsesVisitor {
    fn new(local: Local) -> Self {
        Self { local, first_use: None }
    }

    fn first_use(&self) -> Option<usize> {
        self.first_use
    }
}

impl Visitor<'_> for SuccessorUsesVisitor {
    fn visit_local(&mut self, local: Local, _context: PlaceContext, location: Location) {
        if local == self.local {
            match self.first_use {
                None => self.first_use = Some(location.statement_index),
                Some(first) => self.first_use = Some(first.min(location.statement_index)),
            }
        }
    }
}

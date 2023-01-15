use either::Either;
use rustc_data_structures::graph::dominators::Dominators;
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::middle::resolve_lifetime::Set1;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, TyCtxt};
use rustc_mir_dataflow::impls::borrowed_locals;

use crate::MirPass;

/// Unify locals that copy each other.
///
/// We consider patterns of the form
///   _a = rvalue
///   _b = move? _a
///   _c = move? _a
///   _d = move? _c
/// where each of the locals is only assigned once.
///
/// We want to replace all those locals by `_a`, either copied or moved.
pub struct CopyProp;

impl<'tcx> MirPass<'tcx> for CopyProp {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 4
    }

    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());
        propagate_ssa(tcx, body);
    }
}

fn propagate_ssa<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
    let ssa = SsaLocals::new(tcx, param_env, body);

    let (copy_classes, fully_moved) = compute_copy_classes(&ssa, body);
    debug!(?copy_classes);

    let mut storage_to_remove = BitSet::new_empty(fully_moved.domain_size());
    for (local, &head) in copy_classes.iter_enumerated() {
        if local != head {
            storage_to_remove.insert(head);
            storage_to_remove.insert(local);
        }
    }

    let any_replacement = copy_classes.iter_enumerated().any(|(l, &h)| l != h);

    Replacer { tcx, copy_classes, fully_moved, storage_to_remove }.visit_body_preserves_cfg(body);

    if any_replacement {
        crate::simplify::remove_unused_definitions(body);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LocationExtended {
    Plain(Location),
    Arg,
}

#[derive(Debug)]
struct SsaLocals {
    dominators: Dominators<BasicBlock>,
    /// Assignments to each local.  This defines whether the local is SSA.
    assignments: IndexVec<Local, Set1<LocationExtended>>,
    /// We visit the body in reverse postorder, to ensure each local is assigned before it is used.
    /// We remember the order in which we saw the assignments to compute the SSA values in a single
    /// pass.
    assignment_order: Vec<Local>,
}

impl SsaLocals {
    fn new<'tcx>(tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>, body: &Body<'tcx>) -> SsaLocals {
        let assignment_order = Vec::new();

        let assignments = IndexVec::from_elem(Set1::Empty, &body.local_decls);
        let dominators = body.basic_blocks.dominators();
        let mut this = SsaLocals { assignments, assignment_order, dominators };

        let borrowed = borrowed_locals(body);
        for (local, decl) in body.local_decls.iter_enumerated() {
            if matches!(body.local_kind(local), LocalKind::Arg) {
                this.assignments[local] = Set1::One(LocationExtended::Arg);
            }
            if borrowed.contains(local) && !decl.ty.is_freeze(tcx, param_env) {
                this.assignments[local] = Set1::Many;
            }
        }

        for (bb, data) in traversal::reverse_postorder(body) {
            this.visit_basic_block_data(bb, data);
        }

        for var_debug_info in &body.var_debug_info {
            this.visit_var_debug_info(var_debug_info);
        }

        debug!(?this.assignments);

        this.assignment_order.retain(|&local| matches!(this.assignments[local], Set1::One(_)));
        debug!(?this.assignment_order);

        this
    }
}

impl<'tcx> Visitor<'tcx> for SsaLocals {
    fn visit_local(&mut self, local: Local, ctxt: PlaceContext, loc: Location) {
        match ctxt {
            PlaceContext::MutatingUse(MutatingUseContext::Store) => {
                self.assignments[local].insert(LocationExtended::Plain(loc));
                self.assignment_order.push(local);
            }
            PlaceContext::MutatingUse(_) => self.assignments[local] = Set1::Many,
            // Immutable borrows and AddressOf are taken into account in `SsaLocals::new` by
            // removing non-freeze locals.
            PlaceContext::NonMutatingUse(_) => {
                let set = &mut self.assignments[local];
                let assign_dominates = match *set {
                    Set1::Empty | Set1::Many => false,
                    Set1::One(LocationExtended::Arg) => true,
                    Set1::One(LocationExtended::Plain(assign)) => {
                        assign.dominates(loc, &self.dominators)
                    }
                };
                // We are visiting a use that is not dominated by an assignment.
                // Either there is a cycle involved, or we are reading for uninitialized local.
                // Bail out.
                if !assign_dominates {
                    *set = Set1::Many;
                }
            }
            PlaceContext::NonUse(_) => {}
        }
    }
}

/// Compute the equivalence classes for locals, based on copy statements.
///
/// The returned vector maps each local to the one it copies. In the following case:
///   _a = &mut _0
///   _b = move? _a
///   _c = move? _a
///   _d = move? _c
/// We return the mapping
///   _a => _a // not a copy so, represented by itself
///   _b => _a
///   _c => _a
///   _d => _a // transitively through _c
///
/// This function also returns whether all the `move?` in the pattern are `move` and not copies.
/// A local which is in the bitset can be replaced by `move _a`.  Otherwise, it must be
/// replaced by `copy _a`, as we cannot move multiple times from `_a`.
///
/// If an operand copies `_c`, it must happen before the assignment `_d = _c`, otherwise it is UB.
/// This means that replacing it by a copy of `_a` if ok, since this copy happens before `_c` is
/// moved, and therefore that `_d` is moved.
#[instrument(level = "trace", skip(ssa, body))]
fn compute_copy_classes(
    ssa: &SsaLocals,
    body: &Body<'_>,
) -> (IndexVec<Local, Local>, BitSet<Local>) {
    let mut copies = IndexVec::from_fn_n(|l| l, body.local_decls.len());
    let mut fully_moved = BitSet::new_filled(copies.len());

    for &local in &ssa.assignment_order {
        debug!(?local);

        if local == RETURN_PLACE {
            // `_0` is special, we cannot rename it.
            continue;
        }

        // This is not SSA: mark that we don't know the value.
        debug!(assignments = ?ssa.assignments[local]);
        let Set1::One(LocationExtended::Plain(loc)) = ssa.assignments[local] else { continue };

        // `loc` must point to a direct assignment to `local`.
        let Either::Left(stmt) = body.stmt_at(loc) else { bug!() };
        let Some((_target, rvalue)) = stmt.kind.as_assign() else { bug!() };
        assert_eq!(_target.as_local(), Some(local));

        let (Rvalue::Use(Operand::Copy(place) | Operand::Move(place)) | Rvalue::CopyForDeref(place))
            = rvalue
        else { continue };

        let Some(rhs) = place.as_local() else { continue };
        let Set1::One(_) = ssa.assignments[rhs] else { continue };

        // We visit in `assignment_order`, ie. reverse post-order, so `rhs` has been
        // visited before `local`, and we just have to copy the representing local.
        copies[local] = copies[rhs];

        if let Rvalue::Use(Operand::Copy(_)) | Rvalue::CopyForDeref(_) = rvalue {
            fully_moved.remove(rhs);
        }
    }

    debug!(?copies);

    // Invariant: `copies` must point to the head of an equivalence class.
    #[cfg(debug_assertions)]
    for &head in copies.iter() {
        assert_eq!(copies[head], head);
    }

    meet_copy_equivalence(&copies, &mut fully_moved);

    (copies, fully_moved)
}

/// Make a property uniform on a copy equivalence class by removing elements.
fn meet_copy_equivalence(copies: &IndexVec<Local, Local>, property: &mut BitSet<Local>) {
    // Consolidate to have a local iff all its copies are.
    //
    // `copies` defines equivalence classes between locals.  The `local`s that recursively
    // move/copy the same local all have the same `head`.
    for (local, &head) in copies.iter_enumerated() {
        // If any copy does not have `property`, then the head is not.
        if !property.contains(local) {
            property.remove(head);
        }
    }
    for (local, &head) in copies.iter_enumerated() {
        // If any copy does not have `property`, then the head doesn't either,
        // then no copy has `property`.
        if !property.contains(head) {
            property.remove(local);
        }
    }

    // Verify that we correctly computed equivalence classes.
    #[cfg(debug_assertions)]
    for (local, &head) in copies.iter_enumerated() {
        assert_eq!(property.contains(local), property.contains(head));
    }
}

/// Utility to help performing subtitution of `*pattern` by `target`.
struct Replacer<'tcx> {
    tcx: TyCtxt<'tcx>,
    fully_moved: BitSet<Local>,
    storage_to_remove: BitSet<Local>,
    copy_classes: IndexVec<Local, Local>,
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        *local = self.copy_classes[*local];
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, loc: Location) {
        if let Operand::Move(place) = *operand
            && let Some(local) = place.as_local()
            && !self.fully_moved.contains(local)
        {
            *operand = Operand::Copy(place);
        }
        self.super_operand(operand, loc);
    }

    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, loc: Location) {
        if let StatementKind::StorageLive(l) | StatementKind::StorageDead(l) = stmt.kind
            && self.storage_to_remove.contains(l)
        {
            stmt.make_nop();
        }
        if let StatementKind::Assign(box (ref place, _)) = stmt.kind
            && let Some(l) = place.as_local()
            && self.copy_classes[l] != l
        {
            stmt.make_nop();
        }
        self.super_statement(stmt, loc);
    }
}

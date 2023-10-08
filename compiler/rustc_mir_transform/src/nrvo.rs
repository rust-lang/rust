//! See the docs for [`RenameReturnPlace`].

use rustc_hir::Mutability;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{MutVisitor, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::borrowed_locals;

use crate::MirPass;

/// This pass looks for MIR that always copies the same local into the return place and eliminates
/// the copy by renaming all uses of that local to `_0`.
///
/// This allows LLVM to perform an optimization similar to the named return value optimization
/// (NRVO) that is guaranteed in C++. This avoids a stack allocation and `memcpy` for the
/// relatively common pattern of allocating a buffer on the stack, mutating it, and returning it by
/// value like so:
///
/// ```rust
/// fn foo(init: fn(&mut [u8; 1024])) -> [u8; 1024] {
///     let mut buf = [0; 1024];
///     init(&mut buf);
///     buf
/// }
/// ```
///
/// For now, this pass is very simple and only capable of eliminating a single copy. A more general
/// version of copy propagation, such as the one based on non-overlapping live ranges in [#47954] and
/// [#71003], could yield even more benefits.
///
/// [#47954]: https://github.com/rust-lang/rust/pull/47954
/// [#71003]: https://github.com/rust-lang/rust/pull/71003
pub struct RenameReturnPlace;

impl<'tcx> MirPass<'tcx> for RenameReturnPlace {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        if !tcx.consider_optimizing(|| format!("RenameReturnPlace {def_id:?}")) {
            return;
        }

        let Some(returned_local) = local_eligible_for_nrvo(body) else {
            debug!("`{:?}` was ineligible for NRVO", def_id);
            return;
        };

        debug!(
            "`{:?}` was eligible for NRVO, making {:?} the return place",
            def_id, returned_local
        );

        RenameToReturnPlace { tcx, to_rename: returned_local }.visit_body_preserves_cfg(body);

        // Clean up the `NOP`s we inserted for statements made useless by our renaming.
        for block_data in body.basic_blocks.as_mut_preserves_cfg() {
            block_data.statements.retain(|stmt| stmt.kind != StatementKind::Nop);
        }

        // Overwrite the debuginfo of `_0` with that of the renamed local.
        let (renamed_decl, ret_decl) = body.local_decls.pick2_mut(returned_local, RETURN_PLACE);

        // Sometimes, the return place is assigned a local of a different but coercible type, for
        // example `&mut T` instead of `&T`. Overwriting the `LocalInfo` for the return place means
        // its type may no longer match the return type of its function. This doesn't cause a
        // problem in codegen because these two types are layout-compatible, but may be unexpected.
        debug!("_0: {:?} = {:?}: {:?}", ret_decl.ty, returned_local, renamed_decl.ty);
        ret_decl.clone_from(renamed_decl);

        // The return place is always mutable.
        ret_decl.mutability = Mutability::Mut;
    }
}

/// MIR that is eligible for the NRVO must fulfill two conditions:
///   1. The return place must not be read prior to the `Return` terminator.
///   2. A simple assignment of a whole local to the return place (e.g., `_0 = _1`) must be the
///      only definition of the return place reaching the `Return` terminator.
///
/// If the MIR fulfills both these conditions, this function returns the `Local` that is assigned
/// to the return place along all possible paths through the control-flow graph.
fn local_eligible_for_nrvo(body: &mut Body<'_>) -> Option<Local> {
    if IsReturnPlaceRead::run(body) {
        return None;
    }

    let mut copied_to_return_place = None;
    for block in body.basic_blocks.indices() {
        // Look for blocks with a `Return` terminator.
        if !matches!(body[block].terminator().kind, TerminatorKind::Return) {
            continue;
        }

        // Look for an assignment of a single local to the return place prior to the `Return`.
        let returned_local = find_local_assigned_to_return_place(block, body)?;
        match body.local_kind(returned_local) {
            // FIXME: Can we do this for arguments as well?
            LocalKind::Arg => return None,

            LocalKind::ReturnPointer => bug!("Return place was assigned to itself?"),
            LocalKind::Temp => {}
        }

        // If multiple different locals are copied to the return place. We can't pick a
        // single one to rename.
        if copied_to_return_place.is_some_and(|old| old != returned_local) {
            return None;
        }

        copied_to_return_place = Some(returned_local);
    }

    copied_to_return_place
}

#[instrument(level = "trace", skip(body), ret)]
fn find_local_assigned_to_return_place(start: BasicBlock, body: &mut Body<'_>) -> Option<Local> {
    // The locals that are assigned-to between `return` and `_0 = _rvo_local`.
    let mut assigned_locals = BitSet::new_empty(body.local_decls.len());
    // Whether we have seen an indirect write.
    let mut seen_indirect = false;

    let mut discard_borrowed_locals = |assigned_locals: &mut BitSet<Local>| {
        // We have an indirect assignment to a local between the assignment to `_0 = _rvo`
        // and `return`. This means we may be modifying the RVO local after the assignment.
        // Discard all borrowed locals to be safe.
        if !seen_indirect {
            assigned_locals.union(&borrowed_locals(body));
            // Mark that we have seen an indirect write to avoid recomputing `borrowed_locals`.
            seen_indirect = true;
        }
    };

    // Iterate as long as `block` has exactly one predecessor that we have not yet visited.
    let mut block = start;
    let mut seen_blocks = BitSet::new_empty(body.basic_blocks.len());
    while seen_blocks.insert(block) {
        trace!("Looking for assignments to `_0` in {:?}", block);
        let bbdata = &body.basic_blocks[block];

        let mut vis = DiscardWrites { assigned_locals: &mut assigned_locals, seen_indirect: false };
        vis.visit_terminator(&bbdata.terminator(), body.terminator_loc(block));
        if vis.seen_indirect {
            discard_borrowed_locals(&mut assigned_locals);
        }

        for (statement_index, stmt) in bbdata.statements.iter().enumerate().rev() {
            if let StatementKind::Assign(box (lhs, ref rhs)) = stmt.kind
                && lhs.as_local() == Some(RETURN_PLACE)
                && let Rvalue::Use(rhs) = rhs
                && let Some(rhs) = rhs.place()
                && let Some(rhs) = rhs.as_local()
                && !assigned_locals.contains(rhs)
            {
                return Some(rhs);
            }

            let mut vis =
                DiscardWrites { assigned_locals: &mut assigned_locals, seen_indirect: false };
            vis.visit_statement(stmt, Location { block, statement_index });
            if vis.seen_indirect {
                discard_borrowed_locals(&mut assigned_locals);
            }
        }

        match body.basic_blocks.predecessors()[block].as_slice() {
            &[pred] => block = pred,
            _ => return None,
        }
    }

    None
}

// If this statement is an assignment of an unprojected local to the return place,
// return that local.
fn as_local_assigned_to_return_place(stmt: &Statement<'_>) -> Option<Local> {
    if let StatementKind::Assign(box (lhs, rhs)) = &stmt.kind {
        if lhs.as_local() == Some(RETURN_PLACE) {
            if let Rvalue::Use(Operand::Copy(rhs) | Operand::Move(rhs)) = rhs {
                return rhs.as_local();
            }
        }
    }

    None
}

struct RenameToReturnPlace<'tcx> {
    to_rename: Local,
    tcx: TyCtxt<'tcx>,
}

/// Replaces all uses of `self.to_rename` with `_0`.
impl<'tcx> MutVisitor<'tcx> for RenameToReturnPlace<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, loc: Location) {
        // Remove assignments of the local being replaced to the return place, since it is now the
        // return place:
        //     _0 = _1
        if as_local_assigned_to_return_place(stmt) == Some(self.to_rename) {
            stmt.kind = StatementKind::Nop;
            return;
        }

        // Remove storage annotations for the local being replaced:
        //     StorageLive(_1)
        if let StatementKind::StorageLive(local) | StatementKind::StorageDead(local) = stmt.kind {
            if local == self.to_rename {
                stmt.kind = StatementKind::Nop;
                return;
            }
        }

        self.super_statement(stmt, loc)
    }

    fn visit_local(&mut self, l: &mut Local, _: PlaceContext, _: Location) {
        if *l == self.to_rename {
            *l = RETURN_PLACE;
        }
    }
}

struct IsReturnPlaceRead(bool);

impl IsReturnPlaceRead {
    fn run(body: &Body<'_>) -> bool {
        let mut vis = IsReturnPlaceRead(false);
        vis.visit_body(body);
        vis.0
    }
}

impl<'tcx> Visitor<'tcx> for IsReturnPlaceRead {
    fn visit_local(&mut self, l: Local, ctxt: PlaceContext, _: Location) {
        if l == RETURN_PLACE && ctxt.is_use() && !ctxt.is_place_assignment() {
            self.0 = true;
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, loc: Location) {
        // Ignore the implicit "use" of the return place in a `Return` statement.
        if let TerminatorKind::Return = terminator.kind {
            return;
        }

        self.super_terminator(terminator, loc);
    }
}

struct DiscardWrites<'a> {
    assigned_locals: &'a mut BitSet<Local>,
    seen_indirect: bool,
}

impl<'tcx> Visitor<'tcx> for DiscardWrites<'_> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctxt: PlaceContext, _: Location) {
        match ctxt {
            PlaceContext::MutatingUse(_)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => {
                self.seen_indirect |= place.is_indirect_first_projection();
                self.assigned_locals.insert(place.local);
            }
            PlaceContext::NonMutatingUse(_) | PlaceContext::NonUse(_) => {}
        }
    }
}

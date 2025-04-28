use rustc_hir::{Expr, HirId};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    BasicBlock, Body, InlineAsmOperand, Local, Location, Place, START_BLOCK, StatementKind, TerminatorKind, traversal,
};
use rustc_middle::ty::TyCtxt;

mod possible_borrower;
pub use possible_borrower::PossibleBorrowerMap;

mod possible_origin;

mod transitive_relation;

#[derive(Clone, Debug, Default)]
pub struct LocalUsage {
    /// The locations where the local is used, if any.
    pub local_use_locs: Vec<Location>,
    /// The locations where the local is consumed or mutated, if any.
    pub local_consume_or_mutate_locs: Vec<Location>,
}

pub fn visit_local_usage(locals: &[Local], mir: &Body<'_>, location: Location) -> Option<Vec<LocalUsage>> {
    let init = vec![
        LocalUsage {
            local_use_locs: Vec::new(),
            local_consume_or_mutate_locs: Vec::new(),
        };
        locals.len()
    ];

    traversal::Postorder::new(&mir.basic_blocks, location.block, None)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .try_fold(init, |usage, tbb| {
            let tdata = &mir.basic_blocks[tbb];

            // Give up on loops
            if tdata.terminator().successors().any(|s| s == location.block) {
                return None;
            }

            let mut v = V {
                locals,
                location,
                results: usage,
            };
            v.visit_basic_block_data(tbb, tdata);
            Some(v.results)
        })
}

struct V<'a> {
    locals: &'a [Local],
    location: Location,
    results: Vec<LocalUsage>,
}

impl<'tcx> Visitor<'tcx> for V<'_> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
        if loc.block == self.location.block && loc.statement_index <= self.location.statement_index {
            return;
        }

        let local = place.local;

        for (i, self_local) in self.locals.iter().enumerate() {
            if local == *self_local {
                if !matches!(
                    ctx,
                    PlaceContext::MutatingUse(MutatingUseContext::Drop) | PlaceContext::NonUse(_)
                ) {
                    self.results[i].local_use_locs.push(loc);
                }
                if matches!(
                    ctx,
                    PlaceContext::NonMutatingUse(NonMutatingUseContext::Move | NonMutatingUseContext::Inspect)
                        | PlaceContext::MutatingUse(MutatingUseContext::Borrow)
                ) {
                    self.results[i].local_consume_or_mutate_locs.push(loc);
                }
            }
        }
    }
}

/// Checks if the block is part of a cycle
pub fn block_in_cycle(body: &Body<'_>, block: BasicBlock) -> bool {
    let mut seen = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut to_visit = Vec::with_capacity(body.basic_blocks.len() / 2);

    seen.insert(block);
    let mut next = block;
    loop {
        for succ in body.basic_blocks[next].terminator().successors() {
            if seen.insert(succ) {
                to_visit.push(succ);
            } else if succ == block {
                return true;
            }
        }

        if let Some(x) = to_visit.pop() {
            next = x;
        } else {
            return false;
        }
    }
}

/// Convenience wrapper around `visit_local_usage`.
pub fn used_exactly_once(mir: &Body<'_>, local: Local) -> Option<bool> {
    visit_local_usage(
        &[local],
        mir,
        Location {
            block: START_BLOCK,
            statement_index: 0,
        },
    )
    .map(|mut vec| {
        let LocalUsage { local_use_locs, .. } = vec.remove(0);
        let mut locations = local_use_locs
            .into_iter()
            .filter(|&location| !is_local_assignment(mir, local, location));
        if let Some(location) = locations.next() {
            locations.next().is_none() && !block_in_cycle(mir, location.block)
        } else {
            false
        }
    })
}

/// Returns the `mir::Body` containing the node associated with `hir_id`.
#[allow(clippy::module_name_repetitions)]
pub fn enclosing_mir(tcx: TyCtxt<'_>, hir_id: HirId) -> Option<&Body<'_>> {
    let body_owner_local_def_id = tcx.hir_enclosing_body_owner(hir_id);
    if tcx.hir_body_owner_kind(body_owner_local_def_id).is_fn_or_closure() {
        Some(tcx.optimized_mir(body_owner_local_def_id.to_def_id()))
    } else {
        None
    }
}

/// Tries to determine the `Local` corresponding to `expr`, if any.
/// This function is expensive and should be used sparingly.
pub fn expr_local(tcx: TyCtxt<'_>, expr: &Expr<'_>) -> Option<Local> {
    enclosing_mir(tcx, expr.hir_id).and_then(|mir| {
        mir.local_decls.iter_enumerated().find_map(|(local, local_decl)| {
            if local_decl.source_info.span == expr.span {
                Some(local)
            } else {
                None
            }
        })
    })
}

/// Returns a vector of `mir::Location` where `local` is assigned.
pub fn local_assignments(mir: &Body<'_>, local: Local) -> Vec<Location> {
    let mut locations = Vec::new();
    for (block, data) in mir.basic_blocks.iter_enumerated() {
        for statement_index in 0..=data.statements.len() {
            let location = Location { block, statement_index };
            if is_local_assignment(mir, local, location) {
                locations.push(location);
            }
        }
    }
    locations
}

// `is_local_assignment` is based on `is_place_assignment`:
// https://github.com/rust-lang/rust/blob/b7413511dc85ec01ef4b91785f86614589ac6103/compiler/rustc_middle/src/mir/visit.rs#L1350
fn is_local_assignment(mir: &Body<'_>, local: Local, location: Location) -> bool {
    let Location { block, statement_index } = location;
    let basic_block = &mir.basic_blocks[block];
    if statement_index < basic_block.statements.len() {
        let statement = &basic_block.statements[statement_index];
        if let StatementKind::Assign(box (place, _)) = statement.kind {
            place.as_local() == Some(local)
        } else {
            false
        }
    } else {
        let terminator = basic_block.terminator();
        match &terminator.kind {
            TerminatorKind::Call { destination, .. } => destination.as_local() == Some(local),
            TerminatorKind::InlineAsm { operands, .. } => operands.iter().any(|operand| {
                if let InlineAsmOperand::Out { place: Some(place), .. } = operand {
                    place.as_local() == Some(local)
                } else {
                    false
                }
            }),
            _ => false,
        }
    }
}

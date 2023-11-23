//! Functions dedicated to fact generation for the `-Zpolonius=legacy` datalog implementation.
//!
//! Will be removed in the future, once the in-tree `-Zpolonius=next` implementation reaches feature
//! parity.

use rustc_middle::mir::{Body, LocalKind, Location, START_BLOCK};
use rustc_mir_dataflow::move_paths::{InitKind, InitLocation, MoveData};

use crate::facts::AllFacts;
use crate::location::LocationTable;

/// Emit polonius facts needed for move/init analysis: moves and assignments.
pub(crate) fn emit_move_facts(
    all_facts: &mut AllFacts,
    move_data: &MoveData<'_>,
    location_table: &LocationTable,
    body: &Body<'_>,
) {
    all_facts
        .path_is_var
        .extend(move_data.rev_lookup.iter_locals_enumerated().map(|(l, r)| (r, l)));

    for (child, move_path) in move_data.move_paths.iter_enumerated() {
        if let Some(parent) = move_path.parent {
            all_facts.child_path.push((child, parent));
        }
    }

    let fn_entry_start =
        location_table.start_index(Location { block: START_BLOCK, statement_index: 0 });

    // initialized_at
    for init in move_data.inits.iter() {
        match init.location {
            InitLocation::Statement(location) => {
                let block_data = &body[location.block];
                let is_terminator = location.statement_index == block_data.statements.len();

                if is_terminator && init.kind == InitKind::NonPanicPathOnly {
                    // We are at the terminator of an init that has a panic path,
                    // and where the init should not happen on panic

                    for successor in block_data.terminator().successors() {
                        if body[successor].is_cleanup {
                            continue;
                        }

                        // The initialization happened in (or rather, when arriving at)
                        // the successors, but not in the unwind block.
                        let first_statement = Location { block: successor, statement_index: 0 };
                        all_facts
                            .path_assigned_at_base
                            .push((init.path, location_table.start_index(first_statement)));
                    }
                } else {
                    // In all other cases, the initialization just happens at the
                    // midpoint, like any other effect.
                    all_facts
                        .path_assigned_at_base
                        .push((init.path, location_table.mid_index(location)));
                }
            }
            // Arguments are initialized on function entry
            InitLocation::Argument(local) => {
                assert!(body.local_kind(local) == LocalKind::Arg);
                all_facts.path_assigned_at_base.push((init.path, fn_entry_start));
            }
        }
    }

    for (local, path) in move_data.rev_lookup.iter_locals_enumerated() {
        if body.local_kind(local) != LocalKind::Arg {
            // Non-arguments start out deinitialised; we simulate this with an
            // initial move:
            all_facts.path_moved_at_base.push((path, fn_entry_start));
        }
    }

    // moved_out_at
    // deinitialisation is assumed to always happen!
    all_facts
        .path_moved_at_base
        .extend(move_data.moves.iter().map(|mo| (mo.path, location_table.mid_index(mo.source))));
}

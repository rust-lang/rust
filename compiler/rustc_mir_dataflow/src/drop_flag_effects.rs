use rustc_abi::VariantIdx;
use rustc_middle::mir::{self, Body, Location, Terminator, TerminatorKind};
use tracing::debug;

use super::move_paths::{InitKind, LookupResult, MoveData, MovePathIndex};
use crate::elaborate_drops::DropFlagState;

pub fn move_path_children_matching<'tcx, F>(
    move_data: &MoveData<'tcx>,
    path: MovePathIndex,
    mut cond: F,
) -> Option<MovePathIndex>
where
    F: FnMut(mir::PlaceElem<'tcx>) -> bool,
{
    let mut next_child = move_data.move_paths[path].first_child;
    while let Some(child_index) = next_child {
        let move_path_children = &move_data.move_paths[child_index];
        if let Some(&elem) = move_path_children.place.projection.last() {
            if cond(elem) {
                return Some(child_index);
            }
        }
        next_child = move_path_children.next_sibling;
    }

    None
}

pub fn on_lookup_result_bits<'tcx, F>(
    move_data: &MoveData<'tcx>,
    lookup_result: LookupResult,
    each_child: F,
) where
    F: FnMut(MovePathIndex),
{
    match lookup_result {
        LookupResult::Parent(..) => {
            // access to untracked value - do not touch children
        }
        LookupResult::Exact(e) => on_all_children_bits(move_data, e, each_child),
    }
}

pub fn on_all_children_bits<'tcx, F>(
    move_data: &MoveData<'tcx>,
    move_path_index: MovePathIndex,
    mut each_child: F,
) where
    F: FnMut(MovePathIndex),
{
    fn on_all_children_bits<'tcx, F>(
        move_data: &MoveData<'tcx>,
        move_path_index: MovePathIndex,
        each_child: &mut F,
    ) where
        F: FnMut(MovePathIndex),
    {
        each_child(move_path_index);

        let mut next_child_index = move_data.move_paths[move_path_index].first_child;
        while let Some(child_index) = next_child_index {
            on_all_children_bits(move_data, child_index, each_child);
            next_child_index = move_data.move_paths[child_index].next_sibling;
        }
    }
    on_all_children_bits(move_data, move_path_index, &mut each_child);
}

pub fn drop_flag_effects_for_function_entry<'tcx, F>(
    body: &Body<'tcx>,
    move_data: &MoveData<'tcx>,
    mut callback: F,
) where
    F: FnMut(MovePathIndex, DropFlagState),
{
    for arg in body.args_iter() {
        let place = mir::Place::from(arg);
        let lookup_result = move_data.rev_lookup.find(place.as_ref());
        on_lookup_result_bits(move_data, lookup_result, |mpi| {
            callback(mpi, DropFlagState::Present)
        });
    }
}

pub fn drop_flag_effects_for_location<'tcx, F>(
    body: &Body<'tcx>,
    move_data: &MoveData<'tcx>,
    loc: Location,
    mut callback: F,
) where
    F: FnMut(MovePathIndex, DropFlagState),
{
    debug!("drop_flag_effects_for_location({:?})", loc);

    // first, move out of the RHS
    for mi in &move_data.loc_map[loc] {
        let path = mi.move_path_index(move_data);
        debug!("moving out of path {:?}", move_data.move_paths[path]);

        on_all_children_bits(move_data, path, |mpi| callback(mpi, DropFlagState::Absent))
    }

    // Drop does not count as a move but we should still consider the variable uninitialized.
    if let Some(Terminator { kind: TerminatorKind::Drop { place, .. }, .. }) =
        body.stmt_at(loc).right()
    {
        if let LookupResult::Exact(mpi) = move_data.rev_lookup.find(place.as_ref()) {
            on_all_children_bits(move_data, mpi, |mpi| callback(mpi, DropFlagState::Absent))
        }
    }

    debug!("drop_flag_effects: assignment for location({:?})", loc);

    for_location_inits(move_data, loc, |mpi| callback(mpi, DropFlagState::Present));
}

fn for_location_inits<'tcx, F>(move_data: &MoveData<'tcx>, loc: Location, mut callback: F)
where
    F: FnMut(MovePathIndex),
{
    for ii in &move_data.init_loc_map[loc] {
        let init = move_data.inits[*ii];
        match init.kind {
            InitKind::Deep => {
                let path = init.path;

                on_all_children_bits(move_data, path, &mut callback)
            }
            InitKind::Shallow => {
                let mpi = init.path;
                callback(mpi);
            }
            InitKind::NonPanicPathOnly => (),
        }
    }
}

/// Calls `handle_inactive_variant` for each descendant move path of `enum_place` that contains a
/// `Downcast` to a variant besides the `active_variant`.
///
/// NOTE: If there are no move paths corresponding to an inactive variant,
/// `handle_inactive_variant` will not be called for that variant.
pub(crate) fn on_all_inactive_variants<'tcx>(
    move_data: &MoveData<'tcx>,
    enum_place: mir::Place<'tcx>,
    active_variant: VariantIdx,
    mut handle_inactive_variant: impl FnMut(MovePathIndex),
) {
    let LookupResult::Exact(enum_mpi) = move_data.rev_lookup.find(enum_place.as_ref()) else {
        return;
    };

    let enum_path = &move_data.move_paths[enum_mpi];
    for (variant_mpi, variant_path) in enum_path.children(&move_data.move_paths) {
        // Because of the way we build the `MoveData` tree, each child should have exactly one more
        // projection than `enum_place`. This additional projection must be a downcast since the
        // base is an enum.
        let (downcast, base_proj) = variant_path.place.projection.split_last().unwrap();
        assert_eq!(enum_place.projection.len(), base_proj.len());

        let mir::ProjectionElem::Downcast(_, variant_idx) = *downcast else {
            unreachable!();
        };

        if variant_idx != active_variant {
            on_all_children_bits(move_data, variant_mpi, |mpi| handle_inactive_variant(mpi));
        }
    }
}

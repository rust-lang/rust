use crate::util::elaborate_drops::DropFlagState;
use rustc_middle::mir::{self, Body, Location};
use rustc_middle::ty::{self, TyCtxt};
use rustc_target::abi::VariantIdx;

use super::indexes::MovePathIndex;
use super::move_paths::{InitKind, LookupResult, MoveData};
use super::MoveDataParamEnv;

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

/// When enumerating the child fragments of a path, don't recurse into
/// paths (1.) past arrays, slices, and pointers, nor (2.) into a type
/// that implements `Drop`.
///
/// Places behind references or arrays are not tracked by elaboration
/// and are always assumed to be initialized when accessible. As
/// references and indexes can be reseated, trying to track them can
/// only lead to trouble.
///
/// Places behind ADT's with a Drop impl are not tracked by
/// elaboration since they can never have a drop-flag state that
/// differs from that of the parent with the Drop impl.
///
/// In both cases, the contents can only be accessed if and only if
/// their parents are initialized. This implies for example that there
/// is no need to maintain separate drop flags to track such state.
//
// FIXME: we have to do something for moving slice patterns.
fn place_contents_drop_state_cannot_differ<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    place: mir::Place<'tcx>,
) -> bool {
    let ty = place.ty(body, tcx).ty;
    match ty.kind() {
        ty::Array(..) => {
            debug!(
                "place_contents_drop_state_cannot_differ place: {:?} ty: {:?} => false",
                place, ty
            );
            false
        }
        ty::Slice(..) | ty::Ref(..) | ty::RawPtr(..) => {
            debug!(
                "place_contents_drop_state_cannot_differ place: {:?} ty: {:?} refd => true",
                place, ty
            );
            true
        }
        ty::Adt(def, _) if (def.has_dtor(tcx) && !def.is_box()) || def.is_union() => {
            debug!(
                "place_contents_drop_state_cannot_differ place: {:?} ty: {:?} Drop => true",
                place, ty
            );
            true
        }
        _ => false,
    }
}

pub(crate) fn on_lookup_result_bits<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
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
        LookupResult::Exact(e) => on_all_children_bits(tcx, body, move_data, e, each_child),
    }
}

pub(crate) fn on_all_children_bits<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    move_data: &MoveData<'tcx>,
    move_path_index: MovePathIndex,
    mut each_child: F,
) where
    F: FnMut(MovePathIndex),
{
    fn is_terminal_path<'tcx>(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        move_data: &MoveData<'tcx>,
        path: MovePathIndex,
    ) -> bool {
        place_contents_drop_state_cannot_differ(tcx, body, move_data.move_paths[path].place)
    }

    fn on_all_children_bits<'tcx, F>(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        move_data: &MoveData<'tcx>,
        move_path_index: MovePathIndex,
        each_child: &mut F,
    ) where
        F: FnMut(MovePathIndex),
    {
        each_child(move_path_index);

        if is_terminal_path(tcx, body, move_data, move_path_index) {
            return;
        }

        let mut next_child_index = move_data.move_paths[move_path_index].first_child;
        while let Some(child_index) = next_child_index {
            on_all_children_bits(tcx, body, move_data, child_index, each_child);
            next_child_index = move_data.move_paths[child_index].next_sibling;
        }
    }
    on_all_children_bits(tcx, body, move_data, move_path_index, &mut each_child);
}

pub(crate) fn on_all_drop_children_bits<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    ctxt: &MoveDataParamEnv<'tcx>,
    path: MovePathIndex,
    mut each_child: F,
) where
    F: FnMut(MovePathIndex),
{
    on_all_children_bits(tcx, body, &ctxt.move_data, path, |child| {
        let place = &ctxt.move_data.move_paths[path].place;
        let ty = place.ty(body, tcx).ty;
        debug!("on_all_drop_children_bits({:?}, {:?} : {:?})", path, place, ty);

        let erased_ty = tcx.erase_regions(&ty);
        if erased_ty.needs_drop(tcx, ctxt.param_env) {
            each_child(child);
        } else {
            debug!("on_all_drop_children_bits - skipping")
        }
    })
}

pub(crate) fn drop_flag_effects_for_function_entry<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    ctxt: &MoveDataParamEnv<'tcx>,
    mut callback: F,
) where
    F: FnMut(MovePathIndex, DropFlagState),
{
    let move_data = &ctxt.move_data;
    for arg in body.args_iter() {
        let place = mir::Place::from(arg);
        let lookup_result = move_data.rev_lookup.find(place.as_ref());
        on_lookup_result_bits(tcx, body, move_data, lookup_result, |mpi| {
            callback(mpi, DropFlagState::Present)
        });
    }
}

pub(crate) fn drop_flag_effects_for_location<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    ctxt: &MoveDataParamEnv<'tcx>,
    loc: Location,
    mut callback: F,
) where
    F: FnMut(MovePathIndex, DropFlagState),
{
    let move_data = &ctxt.move_data;
    debug!("drop_flag_effects_for_location({:?})", loc);

    // first, move out of the RHS
    for mi in &move_data.loc_map[loc] {
        let path = mi.move_path_index(move_data);
        debug!("moving out of path {:?}", move_data.move_paths[path]);

        on_all_children_bits(tcx, body, move_data, path, |mpi| callback(mpi, DropFlagState::Absent))
    }

    debug!("drop_flag_effects: assignment for location({:?})", loc);

    for_location_inits(tcx, body, move_data, loc, |mpi| callback(mpi, DropFlagState::Present));
}

pub(crate) fn for_location_inits<'tcx, F>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    move_data: &MoveData<'tcx>,
    loc: Location,
    mut callback: F,
) where
    F: FnMut(MovePathIndex),
{
    for ii in &move_data.init_loc_map[loc] {
        let init = move_data.inits[*ii];
        match init.kind {
            InitKind::Deep => {
                let path = init.path;

                on_all_children_bits(tcx, body, move_data, path, &mut callback)
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
    tcx: TyCtxt<'tcx>,
    body: &mir::Body<'tcx>,
    move_data: &MoveData<'tcx>,
    enum_place: mir::Place<'tcx>,
    active_variant: VariantIdx,
    mut handle_inactive_variant: impl FnMut(MovePathIndex),
) {
    let enum_mpi = match move_data.rev_lookup.find(enum_place.as_ref()) {
        LookupResult::Exact(mpi) => mpi,
        LookupResult::Parent(_) => return,
    };

    let enum_path = &move_data.move_paths[enum_mpi];
    for (variant_mpi, variant_path) in enum_path.children(&move_data.move_paths) {
        // Because of the way we build the `MoveData` tree, each child should have exactly one more
        // projection than `enum_place`. This additional projection must be a downcast since the
        // base is an enum.
        let (downcast, base_proj) = variant_path.place.projection.split_last().unwrap();
        assert_eq!(enum_place.projection.len(), base_proj.len());

        let variant_idx = match *downcast {
            mir::ProjectionElem::Downcast(_, idx) => idx,
            _ => unreachable!(),
        };

        if variant_idx != active_variant {
            on_all_children_bits(tcx, body, move_data, variant_mpi, |mpi| {
                handle_inactive_variant(mpi)
            });
        }
    }
}

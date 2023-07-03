use rustc_index::bit_set::ChunkedBitSet;
use rustc_middle::mir::{Body, TerminatorKind};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt, VariantDef};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{LookupResult, MoveData, MovePathIndex};
use rustc_mir_dataflow::{self, move_path_children_matching, Analysis, MoveDataParamEnv};
use rustc_target::abi::FieldIdx;

use crate::MirPass;

/// Removes `Drop` terminators whose target is known to be uninitialized at
/// that point.
///
/// This is redundant with drop elaboration, but we need to do it prior to const-checking, and
/// running const-checking after drop elaboration makes it optimization dependent, causing issues
/// like [#90770].
///
/// [#90770]: https://github.com/rust-lang/rust/issues/90770
pub struct RemoveUninitDrops;

impl<'tcx> MirPass<'tcx> for RemoveUninitDrops {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env(body.source.def_id());
        let Ok(move_data) = MoveData::gather_moves(body, tcx, param_env) else {
            // We could continue if there are move errors, but there's not much point since our
            // init data isn't complete.
            return;
        };

        let mdpe = MoveDataParamEnv { move_data, param_env };
        let mut maybe_inits = MaybeInitializedPlaces::new(tcx, body, &mdpe)
            .into_engine(tcx, body)
            .pass_name("remove_uninit_drops")
            .iterate_to_fixpoint()
            .into_results_cursor(body);

        let mut to_remove = vec![];
        for (bb, block) in body.basic_blocks.iter_enumerated() {
            let terminator = block.terminator();
            let TerminatorKind::Drop { place, .. } = &terminator.kind
            else { continue };

            maybe_inits.seek_before_primary_effect(body.terminator_loc(bb));

            // If there's no move path for the dropped place, it's probably a `Deref`. Let it alone.
            let LookupResult::Exact(mpi) = mdpe.move_data.rev_lookup.find(place.as_ref()) else {
                continue;
            };

            let should_keep = is_needs_drop_and_init(
                tcx,
                param_env,
                maybe_inits.get(),
                &mdpe.move_data,
                place.ty(body, tcx).ty,
                mpi,
            );
            if !should_keep {
                to_remove.push(bb)
            }
        }

        for bb in to_remove {
            let block = &mut body.basic_blocks_mut()[bb];

            let TerminatorKind::Drop { target, .. }
                = &block.terminator().kind
            else { unreachable!() };

            // Replace block terminator with `Goto`.
            block.terminator_mut().kind = TerminatorKind::Goto { target: *target };
        }
    }
}

fn is_needs_drop_and_init<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    maybe_inits: &ChunkedBitSet<MovePathIndex>,
    move_data: &MoveData<'tcx>,
    ty: Ty<'tcx>,
    mpi: MovePathIndex,
) -> bool {
    // No need to look deeper if the root is definitely uninit or if it has no `Drop` impl.
    if !maybe_inits.contains(mpi) || !ty.needs_drop(tcx, param_env) {
        return false;
    }

    let field_needs_drop_and_init = |(f, f_ty, mpi)| {
        let child = move_path_children_matching(move_data, mpi, |x| x.is_field_to(f));
        let Some(mpi) = child else {
            return Ty::needs_drop(f_ty, tcx, param_env);
        };

        is_needs_drop_and_init(tcx, param_env, maybe_inits, move_data, f_ty, mpi)
    };

    // This pass is only needed for const-checking, so it doesn't handle as many cases as
    // `DropCtxt::open_drop`, since they aren't relevant in a const-context.
    match ty.kind() {
        ty::Adt(adt, substs) => {
            let dont_elaborate = adt.is_union() || adt.is_manually_drop() || adt.has_dtor(tcx);
            if dont_elaborate {
                return true;
            }

            // Look at all our fields, or if we are an enum all our variants and their fields.
            //
            // If a field's projection *is not* present in `MoveData`, it has the same
            // initializedness as its parent (maybe init).
            //
            // If its projection *is* present in `MoveData`, then the field may have been moved
            // from separate from its parent. Recurse.
            adt.variants().iter_enumerated().any(|(vid, variant)| {
                // Enums have multiple variants, which are discriminated with a `Downcast` projection.
                // Structs have a single variant, and don't use a `Downcast` projection.
                let mpi = if adt.is_enum() {
                    let downcast =
                        move_path_children_matching(move_data, mpi, |x| x.is_downcast_to(vid));
                    let Some(dc_mpi) = downcast else {
                        return variant_needs_drop(tcx, param_env, substs, variant);
                    };

                    dc_mpi
                } else {
                    mpi
                };

                variant
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(f, field)| (FieldIdx::from_usize(f), field.ty(tcx, substs), mpi))
                    .any(field_needs_drop_and_init)
            })
        }

        ty::Tuple(fields) => fields
            .iter()
            .enumerate()
            .map(|(f, f_ty)| (FieldIdx::from_usize(f), f_ty, mpi))
            .any(field_needs_drop_and_init),

        _ => true,
    }
}

fn variant_needs_drop<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    substs: SubstsRef<'tcx>,
    variant: &VariantDef,
) -> bool {
    variant.fields.iter().any(|field| {
        let f_ty = field.ty(tcx, substs);
        f_ty.needs_drop(tcx, param_env)
    })
}

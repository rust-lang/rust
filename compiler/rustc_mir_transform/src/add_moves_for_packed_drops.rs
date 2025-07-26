use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use tracing::debug;

use crate::patch::MirPatch;
use crate::util;

/// This pass moves values being dropped that are within a packed
/// struct to a separate local before dropping them, to ensure that
/// they are dropped from an aligned address.
///
/// For example, if we have something like
/// ```ignore (illustrative)
/// #[repr(packed)]
/// struct Foo {
///     dealign: u8,
///     data: Vec<u8>
/// }
///
/// let foo = ...;
/// ```
///
/// We want to call `drop_in_place::<Vec<u8>>` on `data` from an aligned
/// address. This means we can't simply drop `foo.data` directly, because
/// its address is not aligned.
///
/// Instead, we move `foo.data` to a local and drop that:
/// ```ignore (illustrative)
///     storage.live(drop_temp)
///     drop_temp = foo.data;
///     drop(drop_temp) -> next
/// next:
///     storage.dead(drop_temp)
/// ```
///
/// The storage instructions are required to avoid stack space
/// blowup.
pub(super) struct AddMovesForPackedDrops;

impl<'tcx> crate::MirPass<'tcx> for AddMovesForPackedDrops {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("add_moves_for_packed_drops({:?} @ {:?})", body.source, body.span);
        let mut patch = MirPatch::new(body);
        // FIXME(#132279): This is used during the phase transition from analysis
        // to runtime, so we have to manually specify the correct typing mode.
        let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());

        for (bb, data) in body.basic_blocks.iter_enumerated() {
            let loc = Location { block: bb, statement_index: data.statements.len() };
            let terminator = data.terminator();

            match terminator.kind {
                TerminatorKind::Drop { place, .. }
                    if util::is_disaligned(tcx, body, typing_env, place) =>
                {
                    add_move_for_packed_drop(
                        tcx,
                        body,
                        &mut patch,
                        terminator,
                        loc,
                        data.is_cleanup,
                    );
                }
                _ => {}
            }
        }

        patch.apply(body);
    }

    fn is_required(&self) -> bool {
        true
    }
}

fn add_move_for_packed_drop<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    patch: &mut MirPatch<'tcx>,
    terminator: &Terminator<'tcx>,
    loc: Location,
    is_cleanup: bool,
) {
    debug!("add_move_for_packed_drop({:?} @ {:?})", terminator, loc);
    let TerminatorKind::Drop { ref place, target, unwind, replace, drop, async_fut } =
        terminator.kind
    else {
        unreachable!();
    };

    let source_info = terminator.source_info;
    let ty = place.ty(body, tcx).ty;
    let temp = patch.new_temp(ty, source_info.span);

    let storage_dead_block = patch.new_block(BasicBlockData::new_stmts(
        vec![Statement::new(source_info, StatementKind::StorageDead(temp))],
        Some(Terminator { source_info, kind: TerminatorKind::Goto { target } }),
        is_cleanup,
    ));

    patch.add_statement(loc, StatementKind::StorageLive(temp));
    patch.add_assign(loc, Place::from(temp), Rvalue::Use(Operand::Move(*place)));
    patch.patch_terminator(
        loc.block,
        TerminatorKind::Drop {
            place: Place::from(temp),
            target: storage_dead_block,
            unwind,
            replace,
            drop,
            async_fut,
        },
    );
}

//! This pass replaces a drop of a type that does not need dropping, with a goto.
//!
//! When the MIR is built, we check `needs_drop` before emitting a `Drop` for a place. This pass is
//! useful because (unlike MIR building) it runs after type checking, so it can make use of
//! `TypingMode::PostAnalysis` to provide more precise type information, especially about opaque
//! types.
//!
//! When we're optimizing, we also remove calls to `drop_in_place<T>` when `T` isn't `needs_drop`,
//! as those are essentially equivalent to `Drop` terminators. While the compiler doesn't insert
//! them automatically, preferring the built-in instead, they're common in generic code (such as
//! `Vec::truncate`) so removing them from things like inlined `Vec<u8>` is helpful.

use rustc_hir::LangItem;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::{debug, trace};

use super::simplify::simplify_cfg;

pub(super) struct RemoveUnneededDrops;

impl<'tcx> crate::MirPass<'tcx> for RemoveUnneededDrops {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running RemoveUnneededDrops on {:?}", body.source);

        let typing_env = body.typing_env(tcx);
        let mut should_simplify = false;
        for block in body.basic_blocks.as_mut() {
            let terminator = block.terminator_mut();
            let (ty, target) = match terminator.kind {
                TerminatorKind::Drop { place, target, .. } => {
                    (place.ty(&body.local_decls, tcx).ty, target)
                }
                TerminatorKind::Call { ref func, target: Some(target), .. }
                    if tcx.sess.mir_opt_level() > 0
                        && let Some((def_id, generics)) = func.const_fn_def()
                        && tcx.is_lang_item(def_id, LangItem::DropInPlace) =>
                {
                    (generics.type_at(0), target)
                }
                _ => continue,
            };

            if ty.needs_drop(tcx, typing_env) {
                continue;
            }
            debug!("SUCCESS: replacing `drop` with goto({:?})", target);
            terminator.kind = TerminatorKind::Goto { target };
            should_simplify = true;
        }

        // if we applied optimizations, we potentially have some cfg to cleanup to
        // make it easier for further passes
        if should_simplify {
            simplify_cfg(tcx, body);
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

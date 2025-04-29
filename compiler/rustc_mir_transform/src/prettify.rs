//! These two passes provide no value to the compiler, so are off at every level.
//!
//! However, they can be enabled on the command line
//! (`-Zmir-enable-passes=+ReorderBasicBlocks,+ReorderLocals`)
//! to make the MIR easier to read for humans.

use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;

/// Rearranges the basic blocks into a *reverse post-order*.
///
/// Thus after this pass, all the successors of a block are later than it in the
/// `IndexVec`, unless that successor is a back-edge (such as from a loop).
pub(super) struct ReorderBasicBlocks;

impl<'tcx> crate::MirPass<'tcx> for ReorderBasicBlocks {
    fn is_enabled(&self, _session: &Session) -> bool {
        false
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let rpo: IndexVec<BasicBlock, BasicBlock> =
            body.basic_blocks.reverse_postorder().iter().copied().collect();
        if rpo.iter().is_sorted() {
            return;
        }

        let mut updater = BasicBlockUpdater { map: rpo.invert_bijective_mapping(), tcx };
        debug_assert_eq!(updater.map[START_BLOCK], START_BLOCK);
        updater.visit_body(body);

        permute(body.basic_blocks.as_mut(), &updater.map);
    }

    fn is_required(&self) -> bool {
        false
    }
}

/// Rearranges the locals into *use* order.
///
/// Thus after this pass, a local with a smaller [`Location`] where it was first
/// assigned or referenced will have a smaller number.
///
/// (Does not reorder arguments nor the [`RETURN_PLACE`].)
pub(super) struct ReorderLocals;

impl<'tcx> crate::MirPass<'tcx> for ReorderLocals {
    fn is_enabled(&self, _session: &Session) -> bool {
        false
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut finder = LocalFinder {
            map: IndexVec::new(),
            seen: DenseBitSet::new_empty(body.local_decls.len()),
        };

        // We can't reorder the return place or the arguments
        for local in (0..=body.arg_count).map(Local::from_usize) {
            finder.track(local);
        }

        for (bb, bbd) in body.basic_blocks.iter_enumerated() {
            finder.visit_basic_block_data(bb, bbd);
        }

        // Track everything in case there are some locals that we never saw,
        // such as in non-block things like debug info or in non-uses.
        for local in body.local_decls.indices() {
            finder.track(local);
        }

        if finder.map.iter().is_sorted() {
            return;
        }

        let mut updater = LocalUpdater { map: finder.map.invert_bijective_mapping(), tcx };

        for local in (0..=body.arg_count).map(Local::from_usize) {
            debug_assert_eq!(updater.map[local], local);
        }

        updater.visit_body_preserves_cfg(body);

        permute(&mut body.local_decls, &updater.map);
    }

    fn is_required(&self) -> bool {
        false
    }
}

fn permute<I: rustc_index::Idx + Ord, T>(data: &mut IndexVec<I, T>, map: &IndexSlice<I, I>) {
    // FIXME: It would be nice to have a less-awkward way to apply permutations,
    // but I don't know one that exists. `sort_by_cached_key` has logic for it
    // internally, but not in a way that we're allowed to use here.
    let mut enumerated: Vec<_> = std::mem::take(data).into_iter_enumerated().collect();
    enumerated.sort_by_key(|p| map[p.0]);
    *data = enumerated.into_iter().map(|p| p.1).collect();
}

struct BasicBlockUpdater<'tcx> {
    map: IndexVec<BasicBlock, BasicBlock>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for BasicBlockUpdater<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_terminator(&mut self, terminator: &mut Terminator<'tcx>, _location: Location) {
        terminator.successors_mut(|succ| *succ = self.map[*succ]);
    }
}

struct LocalFinder {
    map: IndexVec<Local, Local>,
    seen: DenseBitSet<Local>,
}

impl LocalFinder {
    fn track(&mut self, l: Local) {
        if self.seen.insert(l) {
            self.map.push(l);
        }
    }
}

impl<'tcx> Visitor<'tcx> for LocalFinder {
    fn visit_local(&mut self, l: Local, context: PlaceContext, _location: Location) {
        // Exclude non-uses to keep `StorageLive` from controlling where we put
        // a `Local`, since it might not actually be assigned until much later.
        if context.is_use() {
            self.track(l);
        }
    }
}

struct LocalUpdater<'tcx> {
    map: IndexVec<Local, Local>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for LocalUpdater<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, l: &mut Local, _: PlaceContext, _: Location) {
        *l = self.map[*l];
    }
}

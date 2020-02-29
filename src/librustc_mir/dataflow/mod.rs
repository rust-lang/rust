use rustc::ty;
use rustc_ast::ast::{self, MetaItem};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;
use rustc_span::symbol::{sym, Symbol};

pub(crate) use self::drop_flag_effects::*;
pub use self::framework::{
    visit_results, Analysis, AnalysisDomain, BorrowckFlowState, BorrowckResults, Engine, GenKill,
    GenKillAnalysis, Results, ResultsCursor, ResultsRefCursor, ResultsVisitor,
};
pub use self::impls::{
    borrows::Borrows, DefinitelyInitializedPlaces, EverInitializedPlaces, MaybeBorrowedLocals,
    MaybeInitializedPlaces, MaybeMutBorrowedLocals, MaybeRequiresStorage, MaybeStorageLive,
    MaybeUninitializedPlaces,
};

use self::move_paths::MoveData;

pub mod drop_flag_effects;
pub mod framework;
mod impls;
pub mod move_paths;

pub(crate) mod indexes {
    pub(crate) use super::{
        impls::borrows::BorrowIndex,
        move_paths::{InitIndex, MoveOutIndex, MovePathIndex},
    };
}

pub struct MoveDataParamEnv<'tcx> {
    pub(crate) move_data: MoveData<'tcx>,
    pub(crate) param_env: ty::ParamEnv<'tcx>,
}

pub(crate) fn has_rustc_mir_with(attrs: &[ast::Attribute], name: Symbol) -> Option<MetaItem> {
    for attr in attrs {
        if attr.check_name(sym::rustc_mir) {
            let items = attr.meta_item_list();
            for item in items.iter().flat_map(|l| l.iter()) {
                match item.meta_item() {
                    Some(mi) if mi.check_name(name) => return Some(mi.clone()),
                    _ => continue,
                }
            }
        }
    }
    None
}

/// Parameterization for the precise form of data flow that is used.
///
/// `BottomValue` determines whether the initial entry set for each basic block is empty or full.
/// This also determines the semantics of the lattice `join` operator used to merge dataflow
/// results, since dataflow works by starting at the bottom and moving monotonically to a fixed
/// point.
///
/// This means, for propagation across the graph, that you either want to start at all-zeroes and
/// then use Union as your merge when propagating, or you start at all-ones and then use Intersect
/// as your merge when propagating.
pub trait BottomValue {
    /// Specifies the initial value for each bit in the entry set for each basic block.
    const BOTTOM_VALUE: bool;

    /// Merges `in_set` into `inout_set`, returning `true` if `inout_set` changed.
    ///
    /// It is almost certainly wrong to override this, since it automatically applies
    /// * `inout_set & in_set` if `BOTTOM_VALUE == true`
    /// * `inout_set | in_set` if `BOTTOM_VALUE == false`
    ///
    /// This means that if a bit is not `BOTTOM_VALUE`, it is propagated into all target blocks.
    /// For clarity, the above statement again from a different perspective:
    /// A bit in the block's entry set is `!BOTTOM_VALUE` if *any* predecessor block's bit value is
    /// `!BOTTOM_VALUE`.
    ///
    /// There are situations where you want the opposite behaviour: propagate only if *all*
    /// predecessor blocks's value is `!BOTTOM_VALUE`.
    /// E.g. if you want to know whether a bit is *definitely* set at a specific location. This
    /// means that all code paths leading to the location must have set the bit, instead of any
    /// code path leading there.
    ///
    /// If you want this kind of "definitely set" analysis, you need to
    /// 1. Invert `BOTTOM_VALUE`
    /// 2. Reset the `entry_set` in `start_block_effect` to `!BOTTOM_VALUE`
    /// 3. Override `join` to do the opposite from what it's doing now.
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        if !Self::BOTTOM_VALUE { inout_set.union(in_set) } else { inout_set.intersect(in_set) }
    }
}

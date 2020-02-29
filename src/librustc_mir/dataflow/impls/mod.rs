//! Dataflow analyses are built upon some interpretation of the
//! bitvectors attached to each basic block, represented via a
//! zero-sized structure.

use rustc::mir::{self, Body, Location};
use rustc::ty::layout::VariantIdx;
use rustc::ty::{self, TyCtxt};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;

use super::MoveDataParamEnv;

use crate::util::elaborate_drops::DropFlagState;

use super::move_paths::{HasMoveData, InitIndex, InitKind, LookupResult, MoveData, MovePathIndex};
use super::{AnalysisDomain, BottomValue, GenKill, GenKillAnalysis};

use super::drop_flag_effects_for_function_entry;
use super::drop_flag_effects_for_location;
use super::on_lookup_result_bits;
use crate::dataflow::drop_flag_effects;

mod borrowed_locals;
mod storage_liveness;

pub use self::borrowed_locals::*;
pub use self::storage_liveness::*;

pub(super) mod borrows;

/// `MaybeInitializedPlaces` tracks all places that might be
/// initialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // maybe-init:
///                                            // {}
///     let a = S; let b = S; let c; let d;    // {a, b}
///
///     if pred {
///         drop(a);                           // {   b}
///         b = S;                             // {   b}
///
///     } else {
///         drop(b);                           // {a}
///         d = S;                             // {a,       d}
///
///     }                                      // {a, b,    d}
///
///     c = S;                                 // {a, b, c, d}
/// }
/// ```
///
/// To determine whether a place *must* be initialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeUninitializedPlaces` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeUninitializedPlaces` yields the set of
/// places that would require a dynamic drop-flag at that statement.
pub struct MaybeInitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> MaybeInitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        MaybeInitializedPlaces { tcx, body, mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for MaybeInitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> {
        &self.mdpe.move_data
    }
}

/// `MaybeUninitializedPlaces` tracks all places that might be
/// uninitialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // maybe-uninit:
///                                            // {a, b, c, d}
///     let a = S; let b = S; let c; let d;    // {      c, d}
///
///     if pred {
///         drop(a);                           // {a,    c, d}
///         b = S;                             // {a,    c, d}
///
///     } else {
///         drop(b);                           // {   b, c, d}
///         d = S;                             // {   b, c   }
///
///     }                                      // {a, b, c, d}
///
///     c = S;                                 // {a, b,    d}
/// }
/// ```
///
/// To determine whether a place *must* be uninitialized at a
/// particular control-flow point, one can take the set-difference
/// between this data and the data from `MaybeInitializedPlaces` at the
/// corresponding control-flow point.
///
/// Similarly, at a given `drop` statement, the set-intersection
/// between this data and `MaybeInitializedPlaces` yields the set of
/// places that would require a dynamic drop-flag at that statement.
pub struct MaybeUninitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> MaybeUninitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        MaybeUninitializedPlaces { tcx, body, mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for MaybeUninitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> {
        &self.mdpe.move_data
    }
}

/// `DefinitelyInitializedPlaces` tracks all places that are definitely
/// initialized upon reaching a particular point in the control flow
/// for a function.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // definite-init:
///                                            // {          }
///     let a = S; let b = S; let c; let d;    // {a, b      }
///
///     if pred {
///         drop(a);                           // {   b,     }
///         b = S;                             // {   b,     }
///
///     } else {
///         drop(b);                           // {a,        }
///         d = S;                             // {a,       d}
///
///     }                                      // {          }
///
///     c = S;                                 // {       c  }
/// }
/// ```
///
/// To determine whether a place *may* be uninitialized at a
/// particular control-flow point, one can take the set-complement
/// of this data.
///
/// Similarly, at a given `drop` statement, the set-difference between
/// this data and `MaybeInitializedPlaces` yields the set of places
/// that would require a dynamic drop-flag at that statement.
pub struct DefinitelyInitializedPlaces<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> DefinitelyInitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        DefinitelyInitializedPlaces { tcx, body, mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for DefinitelyInitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> {
        &self.mdpe.move_data
    }
}

/// `EverInitializedPlaces` tracks all places that might have ever been
/// initialized upon reaching a particular point in the control flow
/// for a function, without an intervening `Storage Dead`.
///
/// This dataflow is used to determine if an immutable local variable may
/// be assigned to.
///
/// For example, in code like the following, we have corresponding
/// dataflow information shown in the right-hand comments.
///
/// ```rust
/// struct S;
/// fn foo(pred: bool) {                       // ever-init:
///                                            // {          }
///     let a = S; let b = S; let c; let d;    // {a, b      }
///
///     if pred {
///         drop(a);                           // {a, b,     }
///         b = S;                             // {a, b,     }
///
///     } else {
///         drop(b);                           // {a, b,      }
///         d = S;                             // {a, b,    d }
///
///     }                                      // {a, b,    d }
///
///     c = S;                                 // {a, b, c, d }
/// }
/// ```
pub struct EverInitializedPlaces<'a, 'tcx> {
    #[allow(dead_code)]
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    mdpe: &'a MoveDataParamEnv<'tcx>,
}

impl<'a, 'tcx> EverInitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        EverInitializedPlaces { tcx, body, mdpe }
    }
}

impl<'a, 'tcx> HasMoveData<'tcx> for EverInitializedPlaces<'a, 'tcx> {
    fn move_data(&self) -> &MoveData<'tcx> {
        &self.mdpe.move_data
    }
}

impl<'a, 'tcx> MaybeInitializedPlaces<'a, 'tcx> {
    fn update_bits(
        trans: &mut impl GenKill<MovePathIndex>,
        path: MovePathIndex,
        state: DropFlagState,
    ) {
        match state {
            DropFlagState::Absent => trans.kill(path),
            DropFlagState::Present => trans.gen(path),
        }
    }
}

impl<'a, 'tcx> MaybeUninitializedPlaces<'a, 'tcx> {
    fn update_bits(
        trans: &mut impl GenKill<MovePathIndex>,
        path: MovePathIndex,
        state: DropFlagState,
    ) {
        match state {
            DropFlagState::Absent => trans.gen(path),
            DropFlagState::Present => trans.kill(path),
        }
    }
}

impl<'a, 'tcx> DefinitelyInitializedPlaces<'a, 'tcx> {
    fn update_bits(
        trans: &mut impl GenKill<MovePathIndex>,
        path: MovePathIndex,
        state: DropFlagState,
    ) {
        match state {
            DropFlagState::Absent => trans.kill(path),
            DropFlagState::Present => trans.gen(path),
        }
    }
}

impl<'tcx> AnalysisDomain<'tcx> for MaybeInitializedPlaces<'_, 'tcx> {
    type Idx = MovePathIndex;

    const NAME: &'static str = "maybe_init";

    fn bits_per_block(&self, _: &mir::Body<'tcx>) -> usize {
        self.move_data().move_paths.len()
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, state: &mut BitSet<Self::Idx>) {
        drop_flag_effects_for_function_entry(self.tcx, self.body, self.mdpe, |path, s| {
            assert!(s == DropFlagState::Present);
            state.insert(path);
        });
    }

    fn pretty_print_idx(&self, w: &mut impl std::io::Write, mpi: Self::Idx) -> std::io::Result<()> {
        write!(w, "{}", self.move_data().move_paths[mpi])
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeInitializedPlaces<'_, 'tcx> {
    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.tcx, self.body, self.mdpe, location, |path, s| {
            Self::update_bits(trans, path, s)
        })
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.tcx, self.body, self.mdpe, location, |path, s| {
            Self::update_bits(trans, path, s)
        })
    }

    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        dest_place: &mir::Place<'tcx>,
    ) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 1 (initialized).
        on_lookup_result_bits(
            self.tcx,
            self.body,
            self.move_data(),
            self.move_data().rev_lookup.find(dest_place.as_ref()),
            |mpi| {
                trans.gen(mpi);
            },
        );
    }

    fn discriminant_switch_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        enum_place: &mir::Place<'tcx>,
        _adt: &ty::AdtDef,
        variant: VariantIdx,
    ) {
        let enum_mpi = match self.move_data().rev_lookup.find(enum_place.as_ref()) {
            LookupResult::Exact(mpi) => mpi,
            LookupResult::Parent(_) => return,
        };

        // Kill all move paths that correspond to variants other than this one
        let move_paths = &self.move_data().move_paths;
        let enum_path = &move_paths[enum_mpi];
        for (mpi, variant_path) in enum_path.children(move_paths) {
            trans.kill(mpi);
            match variant_path.place.projection.last().unwrap() {
                mir::ProjectionElem::Downcast(_, idx) if *idx == variant => continue,
                _ => drop_flag_effects::on_all_children_bits(
                    self.tcx,
                    self.body,
                    self.move_data(),
                    mpi,
                    |mpi| trans.kill(mpi),
                ),
            }
        }
    }
}

impl<'tcx> AnalysisDomain<'tcx> for MaybeUninitializedPlaces<'_, 'tcx> {
    type Idx = MovePathIndex;

    const NAME: &'static str = "maybe_uninit";

    fn bits_per_block(&self, _: &mir::Body<'tcx>) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg places
    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut BitSet<Self::Idx>) {
        // set all bits to 1 (uninit) before gathering counterevidence
        assert!(self.bits_per_block(body) == state.domain_size());
        state.insert_all();

        drop_flag_effects_for_function_entry(self.tcx, self.body, self.mdpe, |path, s| {
            assert!(s == DropFlagState::Present);
            state.remove(path);
        });
    }

    fn pretty_print_idx(&self, w: &mut impl std::io::Write, mpi: Self::Idx) -> std::io::Result<()> {
        write!(w, "{}", self.move_data().move_paths[mpi])
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeUninitializedPlaces<'_, 'tcx> {
    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.tcx, self.body, self.mdpe, location, |path, s| {
            Self::update_bits(trans, path, s)
        })
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.tcx, self.body, self.mdpe, location, |path, s| {
            Self::update_bits(trans, path, s)
        })
    }

    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        dest_place: &mir::Place<'tcx>,
    ) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 0 (initialized).
        on_lookup_result_bits(
            self.tcx,
            self.body,
            self.move_data(),
            self.move_data().rev_lookup.find(dest_place.as_ref()),
            |mpi| {
                trans.kill(mpi);
            },
        );
    }
}

impl<'a, 'tcx> AnalysisDomain<'tcx> for DefinitelyInitializedPlaces<'a, 'tcx> {
    type Idx = MovePathIndex;

    const NAME: &'static str = "definite_init";

    fn bits_per_block(&self, _: &mir::Body<'tcx>) -> usize {
        self.move_data().move_paths.len()
    }

    // sets on_entry bits for Arg places
    fn initialize_start_block(&self, _: &mir::Body<'tcx>, state: &mut BitSet<Self::Idx>) {
        state.clear();

        drop_flag_effects_for_function_entry(self.tcx, self.body, self.mdpe, |path, s| {
            assert!(s == DropFlagState::Present);
            state.insert(path);
        });
    }

    fn pretty_print_idx(&self, w: &mut impl std::io::Write, mpi: Self::Idx) -> std::io::Result<()> {
        write!(w, "{}", self.move_data().move_paths[mpi])
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for DefinitelyInitializedPlaces<'_, 'tcx> {
    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.tcx, self.body, self.mdpe, location, |path, s| {
            Self::update_bits(trans, path, s)
        })
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        drop_flag_effects_for_location(self.tcx, self.body, self.mdpe, location, |path, s| {
            Self::update_bits(trans, path, s)
        })
    }

    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        dest_place: &mir::Place<'tcx>,
    ) {
        // when a call returns successfully, that means we need to set
        // the bits for that dest_place to 1 (initialized).
        on_lookup_result_bits(
            self.tcx,
            self.body,
            self.move_data(),
            self.move_data().rev_lookup.find(dest_place.as_ref()),
            |mpi| {
                trans.gen(mpi);
            },
        );
    }
}

impl<'tcx> AnalysisDomain<'tcx> for EverInitializedPlaces<'_, 'tcx> {
    type Idx = InitIndex;

    const NAME: &'static str = "ever_init";

    fn bits_per_block(&self, _: &mir::Body<'tcx>) -> usize {
        self.move_data().inits.len()
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut BitSet<Self::Idx>) {
        for arg_init in 0..body.arg_count {
            state.insert(InitIndex::new(arg_init));
        }
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for EverInitializedPlaces<'_, 'tcx> {
    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &mir::Statement<'tcx>,
        location: Location,
    ) {
        let move_data = self.move_data();
        let init_path_map = &move_data.init_path_map;
        let init_loc_map = &move_data.init_loc_map;
        let rev_lookup = &move_data.rev_lookup;

        debug!(
            "statement {:?} at loc {:?} initializes move_indexes {:?}",
            stmt, location, &init_loc_map[location]
        );
        trans.gen_all(init_loc_map[location].iter().copied());

        match stmt.kind {
            mir::StatementKind::StorageDead(local) => {
                // End inits for StorageDead, so that an immutable variable can
                // be reinitialized on the next iteration of the loop.
                let move_path_index = rev_lookup.find_local(local);
                debug!(
                    "stmt {:?} at loc {:?} clears the ever initialized status of {:?}",
                    stmt, location, &init_path_map[move_path_index]
                );
                trans.kill_all(init_path_map[move_path_index].iter().copied());
            }
            _ => {}
        }
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        let (body, move_data) = (self.body, self.move_data());
        let term = body[location.block].terminator();
        let init_loc_map = &move_data.init_loc_map;
        debug!(
            "terminator {:?} at loc {:?} initializes move_indexes {:?}",
            term, location, &init_loc_map[location]
        );
        trans.gen_all(
            init_loc_map[location]
                .iter()
                .filter(|init_index| {
                    move_data.inits[**init_index].kind != InitKind::NonPanicPathOnly
                })
                .copied(),
        );
    }

    fn call_return_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        block: mir::BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        _dest_place: &mir::Place<'tcx>,
    ) {
        let move_data = self.move_data();
        let init_loc_map = &move_data.init_loc_map;

        let call_loc = self.body.terminator_loc(block);
        for init_index in &init_loc_map[call_loc] {
            trans.gen(*init_index);
        }
    }
}

impl<'a, 'tcx> BottomValue for MaybeInitializedPlaces<'a, 'tcx> {
    /// bottom = uninitialized
    const BOTTOM_VALUE: bool = false;
}

impl<'a, 'tcx> BottomValue for MaybeUninitializedPlaces<'a, 'tcx> {
    /// bottom = initialized (start_block_effect counters this at outset)
    const BOTTOM_VALUE: bool = false;
}

impl<'a, 'tcx> BottomValue for DefinitelyInitializedPlaces<'a, 'tcx> {
    /// bottom = initialized (start_block_effect counters this at outset)
    const BOTTOM_VALUE: bool = true;
}

impl<'a, 'tcx> BottomValue for EverInitializedPlaces<'a, 'tcx> {
    /// bottom = no initialized variables by default
    const BOTTOM_VALUE: bool = false;
}

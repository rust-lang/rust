//! Dataflow analyses are built upon some interpretation of the
//! bitvectors attached to each basic block, represented via a
//! zero-sized structure.

use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;
use rustc_middle::mir::{self, Body, Location};
use rustc_middle::ty::{self, TyCtxt};

use crate::drop_flag_effects;
use crate::drop_flag_effects_for_function_entry;
use crate::drop_flag_effects_for_location;
use crate::elaborate_drops::DropFlagState;
use crate::framework::SwitchIntEdgeEffects;
use crate::move_paths::{HasMoveData, InitIndex, InitKind, MoveData, MovePathIndex};
use crate::on_lookup_result_bits;
use crate::MoveDataParamEnv;
use crate::{lattice, AnalysisDomain, GenKill, GenKillAnalysis};

mod borrowed_locals;
mod init_locals;
mod liveness;
mod storage_liveness;

pub use self::borrowed_locals::MaybeBorrowedLocals;
pub use self::init_locals::MaybeInitializedLocals;
pub use self::liveness::MaybeLiveLocals;
pub use self::storage_liveness::{MaybeRequiresStorage, MaybeStorageLive};

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

    mark_inactive_variants_as_uninit: bool,
}

impl<'a, 'tcx> MaybeUninitializedPlaces<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'a Body<'tcx>, mdpe: &'a MoveDataParamEnv<'tcx>) -> Self {
        MaybeUninitializedPlaces { tcx, body, mdpe, mark_inactive_variants_as_uninit: false }
    }

    /// Causes inactive enum variants to be marked as "maybe uninitialized" after a switch on an
    /// enum discriminant.
    ///
    /// This is correct in a vacuum but is not the default because it causes problems in the borrow
    /// checker, where this information gets propagated along `FakeEdge`s.
    pub fn mark_inactive_variants_as_uninit(mut self) -> Self {
        self.mark_inactive_variants_as_uninit = true;
        self
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
/// for a function, without an intervening `StorageDead`.
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
    type Domain = BitSet<MovePathIndex>;
    const NAME: &'static str = "maybe_init";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = uninitialized
        BitSet::new_empty(self.move_data().move_paths.len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, state: &mut Self::Domain) {
        drop_flag_effects_for_function_entry(self.tcx, self.body, self.mdpe, |path, s| {
            assert!(s == DropFlagState::Present);
            state.insert(path);
        });
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeInitializedPlaces<'_, 'tcx> {
    type Idx = MovePathIndex;

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
        dest_place: mir::Place<'tcx>,
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

    fn switch_int_edge_effects<G: GenKill<Self::Idx>>(
        &self,
        block: mir::BasicBlock,
        discr: &mir::Operand<'tcx>,
        edge_effects: &mut impl SwitchIntEdgeEffects<G>,
    ) {
        if !self.tcx.sess.opts.debugging_opts.precise_enum_drop_elaboration {
            return;
        }

        let enum_ = discr.place().and_then(|discr| {
            switch_on_enum_discriminant(self.tcx, &self.body, &self.body[block], discr)
        });

        let (enum_place, enum_def) = match enum_ {
            Some(x) => x,
            None => return,
        };

        let mut discriminants = enum_def.discriminants(self.tcx);
        edge_effects.apply(|trans, edge| {
            let value = match edge.value {
                Some(x) => x,
                None => return,
            };

            // MIR building adds discriminants to the `values` array in the same order as they
            // are yielded by `AdtDef::discriminants`. We rely on this to match each
            // discriminant in `values` to its corresponding variant in linear time.
            let (variant, _) = discriminants
                .find(|&(_, discr)| discr.val == value)
                .expect("Order of `AdtDef::discriminants` differed from `SwitchInt::values`");

            // Kill all move paths that correspond to variants we know to be inactive along this
            // particular outgoing edge of a `SwitchInt`.
            drop_flag_effects::on_all_inactive_variants(
                self.tcx,
                self.body,
                self.move_data(),
                enum_place,
                variant,
                |mpi| trans.kill(mpi),
            );
        });
    }
}

impl<'tcx> AnalysisDomain<'tcx> for MaybeUninitializedPlaces<'_, 'tcx> {
    type Domain = BitSet<MovePathIndex>;

    const NAME: &'static str = "maybe_uninit";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = initialized (start_block_effect counters this at outset)
        BitSet::new_empty(self.move_data().move_paths.len())
    }

    // sets on_entry bits for Arg places
    fn initialize_start_block(&self, _: &mir::Body<'tcx>, state: &mut Self::Domain) {
        // set all bits to 1 (uninit) before gathering counterevidence
        state.insert_all();

        drop_flag_effects_for_function_entry(self.tcx, self.body, self.mdpe, |path, s| {
            assert!(s == DropFlagState::Present);
            state.remove(path);
        });
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for MaybeUninitializedPlaces<'_, 'tcx> {
    type Idx = MovePathIndex;

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
        dest_place: mir::Place<'tcx>,
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

    fn switch_int_edge_effects<G: GenKill<Self::Idx>>(
        &self,
        block: mir::BasicBlock,
        discr: &mir::Operand<'tcx>,
        edge_effects: &mut impl SwitchIntEdgeEffects<G>,
    ) {
        if !self.tcx.sess.opts.debugging_opts.precise_enum_drop_elaboration {
            return;
        }

        if !self.mark_inactive_variants_as_uninit {
            return;
        }

        let enum_ = discr.place().and_then(|discr| {
            switch_on_enum_discriminant(self.tcx, &self.body, &self.body[block], discr)
        });

        let (enum_place, enum_def) = match enum_ {
            Some(x) => x,
            None => return,
        };

        let mut discriminants = enum_def.discriminants(self.tcx);
        edge_effects.apply(|trans, edge| {
            let value = match edge.value {
                Some(x) => x,
                None => return,
            };

            // MIR building adds discriminants to the `values` array in the same order as they
            // are yielded by `AdtDef::discriminants`. We rely on this to match each
            // discriminant in `values` to its corresponding variant in linear time.
            let (variant, _) = discriminants
                .find(|&(_, discr)| discr.val == value)
                .expect("Order of `AdtDef::discriminants` differed from `SwitchInt::values`");

            // Mark all move paths that correspond to variants other than this one as maybe
            // uninitialized (in reality, they are *definitely* uninitialized).
            drop_flag_effects::on_all_inactive_variants(
                self.tcx,
                self.body,
                self.move_data(),
                enum_place,
                variant,
                |mpi| trans.gen(mpi),
            );
        });
    }
}

impl<'a, 'tcx> AnalysisDomain<'tcx> for DefinitelyInitializedPlaces<'a, 'tcx> {
    /// Use set intersection as the join operator.
    type Domain = lattice::Dual<BitSet<MovePathIndex>>;

    const NAME: &'static str = "definite_init";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = initialized (start_block_effect counters this at outset)
        lattice::Dual(BitSet::new_filled(self.move_data().move_paths.len()))
    }

    // sets on_entry bits for Arg places
    fn initialize_start_block(&self, _: &mir::Body<'tcx>, state: &mut Self::Domain) {
        state.0.clear();

        drop_flag_effects_for_function_entry(self.tcx, self.body, self.mdpe, |path, s| {
            assert!(s == DropFlagState::Present);
            state.0.insert(path);
        });
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for DefinitelyInitializedPlaces<'_, 'tcx> {
    type Idx = MovePathIndex;

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
        dest_place: mir::Place<'tcx>,
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
    type Domain = BitSet<InitIndex>;

    const NAME: &'static str = "ever_init";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = no initialized variables by default
        BitSet::new_empty(self.move_data().inits.len())
    }

    fn initialize_start_block(&self, body: &mir::Body<'tcx>, state: &mut Self::Domain) {
        for arg_init in 0..body.arg_count {
            state.insert(InitIndex::new(arg_init));
        }
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for EverInitializedPlaces<'_, 'tcx> {
    type Idx = InitIndex;

    #[instrument(skip(self, trans), level = "debug")]
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

        debug!("initializes move_indexes {:?}", &init_loc_map[location]);
        trans.gen_all(init_loc_map[location].iter().copied());

        if let mir::StatementKind::StorageDead(local) = stmt.kind {
            // End inits for StorageDead, so that an immutable variable can
            // be reinitialized on the next iteration of the loop.
            let move_path_index = rev_lookup.find_local(local);
            debug!("clears the ever initialized status of {:?}", init_path_map[move_path_index]);
            trans.kill_all(init_path_map[move_path_index].iter().copied());
        }
    }

    #[instrument(skip(self, trans, _terminator), level = "debug")]
    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        let (body, move_data) = (self.body, self.move_data());
        let term = body[location.block].terminator();
        let init_loc_map = &move_data.init_loc_map;
        debug!(?term);
        debug!("initializes move_indexes {:?}", init_loc_map[location]);
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
        _dest_place: mir::Place<'tcx>,
    ) {
        let move_data = self.move_data();
        let init_loc_map = &move_data.init_loc_map;

        let call_loc = self.body.terminator_loc(block);
        for init_index in &init_loc_map[call_loc] {
            trans.gen(*init_index);
        }
    }
}

/// Inspect a `SwitchInt`-terminated basic block to see if the condition of that `SwitchInt` is
/// an enum discriminant.
///
/// We expect such blocks to have a call to `discriminant` as their last statement like so:
///
/// ```text
/// ...
/// _42 = discriminant(_1)
/// SwitchInt(_42, ..)
/// ```
///
/// If the basic block matches this pattern, this function returns the place corresponding to the
/// enum (`_1` in the example above) as well as the `AdtDef` of that enum.
fn switch_on_enum_discriminant(
    tcx: TyCtxt<'tcx>,
    body: &'mir mir::Body<'tcx>,
    block: &'mir mir::BasicBlockData<'tcx>,
    switch_on: mir::Place<'tcx>,
) -> Option<(mir::Place<'tcx>, &'tcx ty::AdtDef)> {
    match block.statements.last().map(|stmt| &stmt.kind) {
        Some(mir::StatementKind::Assign(box (lhs, mir::Rvalue::Discriminant(discriminated))))
            if *lhs == switch_on =>
        {
            match &discriminated.ty(body, tcx).ty.kind() {
                ty::Adt(def, _) => Some((*discriminated, def)),

                // `Rvalue::Discriminant` is also used to get the active yield point for a
                // generator, but we do not need edge-specific effects in that case. This may
                // change in the future.
                ty::Generator(..) => None,

                t => bug!("`discriminant` called on unexpected type {:?}", t),
            }
        }

        _ => None,
    }
}

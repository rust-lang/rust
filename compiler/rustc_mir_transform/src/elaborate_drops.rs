use std::fmt;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_mir_dataflow::impls::{MaybeInitializedPlaces, MaybeUninitializedPlaces};
use rustc_mir_dataflow::move_paths::{LookupResult, MoveData, MovePathIndex};
use rustc_mir_dataflow::{
    Analysis, DropFlagState, MoveDataTypingEnv, ResultsCursor, on_all_children_bits,
    on_lookup_result_bits,
};
use rustc_span::Span;
use tracing::{debug, instrument};

use crate::deref_separator::deref_finder;
use crate::elaborate_drop::{DropElaborator, DropFlagMode, DropStyle, Unwind, elaborate_drop};
use crate::patch::MirPatch;

/// During MIR building, Drop terminators are inserted in every place where a drop may occur.
/// However, in this phase, the presence of these terminators does not guarantee that a destructor
/// will run, as the target of the drop may be uninitialized.
/// In general, the compiler cannot determine at compile time whether a destructor will run or not.
///
/// At a high level, this pass refines Drop to only run the destructor if the
/// target is initialized. The way this is achieved is by inserting drop flags for every variable
/// that may be dropped, and then using those flags to determine whether a destructor should run.
/// Once this is complete, Drop terminators in the MIR correspond to a call to the "drop glue" or
/// "drop shim" for the type of the dropped place.
///
/// This pass relies on dropped places having an associated move path, which is then used to
/// determine the initialization status of the place and its descendants.
/// It's worth noting that a MIR containing a Drop without an associated move path is probably ill
/// formed, as it would allow running a destructor on a place behind a reference:
///
/// ```text
/// fn drop_term<T>(t: &mut T) {
///     mir! {
///         {
///             Drop(*t, exit)
///         }
///         exit = {
///             Return()
///         }
///     }
/// }
/// ```
pub(super) struct ElaborateDrops;

impl<'tcx> crate::MirPass<'tcx> for ElaborateDrops {
    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("elaborate_drops({:?} @ {:?})", body.source, body.span);
        // FIXME(#132279): This is used during the phase transition from analysis
        // to runtime, so we have to manually specify the correct typing mode.
        let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());
        // For types that do not need dropping, the behaviour is trivial. So we only need to track
        // init/uninit for types that do need dropping.
        let move_data = MoveData::gather_moves(body, tcx, |ty| ty.needs_drop(tcx, typing_env));
        let elaborate_patch = {
            let env = MoveDataTypingEnv { move_data, typing_env };

            let mut inits = MaybeInitializedPlaces::new(tcx, body, &env.move_data)
                .exclude_inactive_in_otherwise()
                .skipping_unreachable_unwind()
                .iterate_to_fixpoint(tcx, body, Some("elaborate_drops"))
                .into_results_cursor(body);
            let dead_unwinds = compute_dead_unwinds(body, &mut inits);

            let uninits = MaybeUninitializedPlaces::new(tcx, body, &env.move_data)
                .include_inactive_in_otherwise()
                .mark_inactive_variants_as_uninit()
                .skipping_unreachable_unwind(dead_unwinds)
                .iterate_to_fixpoint(tcx, body, Some("elaborate_drops"))
                .into_results_cursor(body);

            let drop_flags = IndexVec::from_elem(None, &env.move_data.move_paths);
            ElaborateDropsCtxt {
                tcx,
                body,
                env: &env,
                init_data: InitializationData { inits, uninits },
                drop_flags,
                patch: MirPatch::new(body),
            }
            .elaborate()
        };
        elaborate_patch.apply(body);
        deref_finder(tcx, body);
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// Records unwind edges which are known to be unreachable, because they are in `drop` terminators
/// that can't drop anything.
#[instrument(level = "trace", skip(body, flow_inits), ret)]
fn compute_dead_unwinds<'a, 'tcx>(
    body: &'a Body<'tcx>,
    flow_inits: &mut ResultsCursor<'a, 'tcx, MaybeInitializedPlaces<'a, 'tcx>>,
) -> DenseBitSet<BasicBlock> {
    // We only need to do this pass once, because unwind edges can only
    // reach cleanup blocks, which can't have unwind edges themselves.
    let mut dead_unwinds = DenseBitSet::new_empty(body.basic_blocks.len());
    for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::Drop { place, unwind: UnwindAction::Cleanup(_), .. } =
            bb_data.terminator().kind
        else {
            continue;
        };

        flow_inits.seek_before_primary_effect(body.terminator_loc(bb));
        if flow_inits.analysis().is_unwind_dead(place, flow_inits.get()) {
            dead_unwinds.insert(bb);
        }
    }

    dead_unwinds
}

struct InitializationData<'a, 'tcx> {
    inits: ResultsCursor<'a, 'tcx, MaybeInitializedPlaces<'a, 'tcx>>,
    uninits: ResultsCursor<'a, 'tcx, MaybeUninitializedPlaces<'a, 'tcx>>,
}

impl InitializationData<'_, '_> {
    fn seek_before(&mut self, loc: Location) {
        self.inits.seek_before_primary_effect(loc);
        self.uninits.seek_before_primary_effect(loc);
    }

    fn maybe_init_uninit(&self, path: MovePathIndex) -> (bool, bool) {
        (self.inits.get().contains(path), self.uninits.get().contains(path))
    }
}

impl<'a, 'tcx> DropElaborator<'a, 'tcx> for ElaborateDropsCtxt<'a, 'tcx> {
    type Path = MovePathIndex;

    fn patch_ref(&self) -> &MirPatch<'tcx> {
        &self.patch
    }

    fn patch(&mut self) -> &mut MirPatch<'tcx> {
        &mut self.patch
    }

    fn body(&self) -> &'a Body<'tcx> {
        self.body
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.env.typing_env
    }

    fn allow_async_drops(&self) -> bool {
        true
    }

    fn terminator_loc(&self, bb: BasicBlock) -> Location {
        self.patch.terminator_loc(self.body, bb)
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn drop_style(&self, path: Self::Path, mode: DropFlagMode) -> DropStyle {
        let ((maybe_init, maybe_uninit), multipart) = match mode {
            DropFlagMode::Shallow => (self.init_data.maybe_init_uninit(path), false),
            DropFlagMode::Deep => {
                let mut some_maybe_init = false;
                let mut some_maybe_uninit = false;
                let mut children_count = 0;
                on_all_children_bits(self.move_data(), path, |child| {
                    let (maybe_init, maybe_uninit) = self.init_data.maybe_init_uninit(child);
                    debug!("elaborate_drop: state({:?}) = {:?}", child, (maybe_init, maybe_uninit));
                    some_maybe_init |= maybe_init;
                    some_maybe_uninit |= maybe_uninit;
                    children_count += 1;
                });
                ((some_maybe_init, some_maybe_uninit), children_count != 1)
            }
        };
        match (maybe_init, maybe_uninit, multipart) {
            (false, _, _) => DropStyle::Dead,
            (true, false, _) => DropStyle::Static,
            (true, true, false) => DropStyle::Conditional,
            (true, true, true) => DropStyle::Open,
        }
    }

    fn clear_drop_flag(&mut self, loc: Location, path: Self::Path, mode: DropFlagMode) {
        match mode {
            DropFlagMode::Shallow => {
                self.set_drop_flag(loc, path, DropFlagState::Absent);
            }
            DropFlagMode::Deep => {
                on_all_children_bits(self.move_data(), path, |child| {
                    self.set_drop_flag(loc, child, DropFlagState::Absent)
                });
            }
        }
    }

    fn field_subpath(&self, path: Self::Path, field: FieldIdx) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.move_data(), path, |e| match e {
            ProjectionElem::Field(idx, _) => idx == field,
            _ => false,
        })
    }

    fn array_subpath(&self, path: Self::Path, index: u64, size: u64) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.move_data(), path, |e| match e {
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                debug_assert!(size == min_length, "min_length should be exact for arrays");
                assert!(!from_end, "from_end should not be used for array element ConstantIndex");
                offset == index
            }
            _ => false,
        })
    }

    fn deref_subpath(&self, path: Self::Path) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.move_data(), path, |e| {
            e == ProjectionElem::Deref
        })
    }

    fn downcast_subpath(&self, path: Self::Path, variant: VariantIdx) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.move_data(), path, |e| match e {
            ProjectionElem::Downcast(_, idx) => idx == variant,
            _ => false,
        })
    }

    fn get_drop_flag(&mut self, path: Self::Path) -> Option<Operand<'tcx>> {
        self.drop_flag(path).map(Operand::Copy)
    }
}

struct ElaborateDropsCtxt<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    env: &'a MoveDataTypingEnv<'tcx>,
    init_data: InitializationData<'a, 'tcx>,
    drop_flags: IndexVec<MovePathIndex, Option<Local>>,
    patch: MirPatch<'tcx>,
}

impl fmt::Debug for ElaborateDropsCtxt<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElaborateDropsCtxt").finish_non_exhaustive()
    }
}

impl<'a, 'tcx> ElaborateDropsCtxt<'a, 'tcx> {
    fn move_data(&self) -> &'a MoveData<'tcx> {
        &self.env.move_data
    }

    fn create_drop_flag(&mut self, index: MovePathIndex, span: Span) {
        let patch = &mut self.patch;
        debug!("create_drop_flag({:?})", self.body.span);
        self.drop_flags[index].get_or_insert_with(|| patch.new_temp(self.tcx.types.bool, span));
    }

    fn drop_flag(&mut self, index: MovePathIndex) -> Option<Place<'tcx>> {
        self.drop_flags[index].map(Place::from)
    }

    /// create a patch that elaborates all drops in the input
    /// MIR.
    fn elaborate(mut self) -> MirPatch<'tcx> {
        self.collect_drop_flags();

        self.elaborate_drops();

        self.drop_flags_on_init();
        self.drop_flags_for_fn_rets();
        self.drop_flags_for_args();
        self.drop_flags_for_locs();

        self.patch
    }

    fn collect_drop_flags(&mut self) {
        for (bb, data) in self.body.basic_blocks.iter_enumerated() {
            let terminator = data.terminator();
            let TerminatorKind::Drop { ref place, .. } = terminator.kind else { continue };

            let path = self.move_data().rev_lookup.find(place.as_ref());
            debug!("collect_drop_flags: {:?}, place {:?} ({:?})", bb, place, path);

            match path {
                LookupResult::Exact(path) => {
                    self.init_data.seek_before(self.body.terminator_loc(bb));
                    on_all_children_bits(self.move_data(), path, |child| {
                        let (maybe_init, maybe_uninit) = self.init_data.maybe_init_uninit(child);
                        debug!(
                            "collect_drop_flags: collecting {:?} from {:?}@{:?} - {:?}",
                            child,
                            place,
                            path,
                            (maybe_init, maybe_uninit)
                        );
                        if maybe_init && maybe_uninit {
                            self.create_drop_flag(child, terminator.source_info.span)
                        }
                    });
                }
                LookupResult::Parent(None) => {}
                LookupResult::Parent(Some(parent)) => {
                    if self.body.local_decls[place.local].is_deref_temp() {
                        continue;
                    }

                    self.init_data.seek_before(self.body.terminator_loc(bb));
                    let (_maybe_init, maybe_uninit) = self.init_data.maybe_init_uninit(parent);
                    if maybe_uninit {
                        self.tcx.dcx().span_delayed_bug(
                            terminator.source_info.span,
                            format!(
                                "drop of untracked, uninitialized value {bb:?}, place {place:?} ({path:?})"
                            ),
                        );
                    }
                }
            };
        }
    }

    fn elaborate_drops(&mut self) {
        // This function should mirror what `collect_drop_flags` does.
        for (bb, data) in self.body.basic_blocks.iter_enumerated() {
            let terminator = data.terminator();
            let TerminatorKind::Drop { place, target, unwind, replace, drop, async_fut: _ } =
                terminator.kind
            else {
                continue;
            };

            // This place does not need dropping. It does not have an associated move-path, so the
            // match below will conservatively keep an unconditional drop. As that drop is useless,
            // just remove it here and now.
            if !place
                .ty(&self.body.local_decls, self.tcx)
                .ty
                .needs_drop(self.tcx, self.typing_env())
            {
                self.patch.patch_terminator(bb, TerminatorKind::Goto { target });
                continue;
            }

            let path = self.move_data().rev_lookup.find(place.as_ref());
            match path {
                LookupResult::Exact(path) => {
                    let unwind = match unwind {
                        _ if data.is_cleanup => Unwind::InCleanup,
                        UnwindAction::Cleanup(cleanup) => Unwind::To(cleanup),
                        UnwindAction::Continue => Unwind::To(self.patch.resume_block()),
                        UnwindAction::Unreachable => {
                            Unwind::To(self.patch.unreachable_cleanup_block())
                        }
                        UnwindAction::Terminate(reason) => {
                            debug_assert_ne!(
                                reason,
                                UnwindTerminateReason::InCleanup,
                                "we are not in a cleanup block, InCleanup reason should be impossible"
                            );
                            Unwind::To(self.patch.terminate_block(reason))
                        }
                    };
                    self.init_data.seek_before(self.body.terminator_loc(bb));
                    elaborate_drop(
                        self,
                        terminator.source_info,
                        place,
                        path,
                        target,
                        unwind,
                        bb,
                        drop,
                    )
                }
                LookupResult::Parent(None) => {}
                LookupResult::Parent(Some(_)) => {
                    if !replace {
                        self.tcx.dcx().span_bug(
                            terminator.source_info.span,
                            format!("drop of untracked value {bb:?}"),
                        );
                    }
                    // A drop and replace behind a pointer/array/whatever.
                    // The borrow checker requires that these locations are initialized before the
                    // assignment, so we just leave an unconditional drop.
                    assert!(!data.is_cleanup);
                }
            }
        }
    }

    fn constant_bool(&self, span: Span, val: bool) -> Rvalue<'tcx> {
        Rvalue::Use(Operand::Constant(Box::new(ConstOperand {
            span,
            user_ty: None,
            const_: Const::from_bool(self.tcx, val),
        })))
    }

    fn set_drop_flag(&mut self, loc: Location, path: MovePathIndex, val: DropFlagState) {
        if let Some(flag) = self.drop_flags[path] {
            let span = self.patch.source_info_for_location(self.body, loc).span;
            let val = self.constant_bool(span, val.value());
            self.patch.add_assign(loc, Place::from(flag), val);
        }
    }

    fn drop_flags_on_init(&mut self) {
        let loc = Location::START;
        let span = self.patch.source_info_for_location(self.body, loc).span;
        let false_ = self.constant_bool(span, false);
        for flag in self.drop_flags.iter().flatten() {
            self.patch.add_assign(loc, Place::from(*flag), false_.clone());
        }
    }

    fn drop_flags_for_fn_rets(&mut self) {
        for (bb, data) in self.body.basic_blocks.iter_enumerated() {
            if let TerminatorKind::Call {
                destination,
                target: Some(tgt),
                unwind: UnwindAction::Cleanup(_),
                ..
            } = data.terminator().kind
            {
                assert!(!self.patch.is_term_patched(bb));

                let loc = Location { block: tgt, statement_index: 0 };
                let path = self.move_data().rev_lookup.find(destination.as_ref());
                on_lookup_result_bits(self.move_data(), path, |child| {
                    self.set_drop_flag(loc, child, DropFlagState::Present)
                });
            }
        }
    }

    fn drop_flags_for_args(&mut self) {
        let loc = Location::START;
        rustc_mir_dataflow::drop_flag_effects_for_function_entry(
            self.body,
            &self.env.move_data,
            |path, ds| {
                self.set_drop_flag(loc, path, ds);
            },
        )
    }

    fn drop_flags_for_locs(&mut self) {
        // We intentionally iterate only over the *old* basic blocks.
        //
        // Basic blocks created by drop elaboration update their
        // drop flags by themselves, to avoid the drop flags being
        // clobbered before they are read.

        for (bb, data) in self.body.basic_blocks.iter_enumerated() {
            debug!("drop_flags_for_locs({:?})", data);
            for i in 0..(data.statements.len() + 1) {
                debug!("drop_flag_for_locs: stmt {}", i);
                if i == data.statements.len() {
                    match data.terminator().kind {
                        TerminatorKind::Drop { .. } => {
                            // drop elaboration should handle that by itself
                            continue;
                        }
                        TerminatorKind::UnwindResume => {
                            // It is possible for `Resume` to be patched
                            // (in particular it can be patched to be replaced with
                            // a Goto; see `MirPatch::new`).
                        }
                        _ => {
                            assert!(!self.patch.is_term_patched(bb));
                        }
                    }
                }
                let loc = Location { block: bb, statement_index: i };
                rustc_mir_dataflow::drop_flag_effects_for_location(
                    self.body,
                    &self.env.move_data,
                    loc,
                    |path, ds| self.set_drop_flag(loc, path, ds),
                )
            }

            // There may be a critical edge after this call,
            // so mark the return as initialized *before* the
            // call.
            if let TerminatorKind::Call {
                destination,
                target: Some(_),
                unwind:
                    UnwindAction::Continue | UnwindAction::Unreachable | UnwindAction::Terminate(_),
                ..
            } = data.terminator().kind
            {
                assert!(!self.patch.is_term_patched(bb));

                let loc = Location { block: bb, statement_index: data.statements.len() };
                let path = self.move_data().rev_lookup.find(destination.as_ref());
                on_lookup_result_bits(self.move_data(), path, |child| {
                    self.set_drop_flag(loc, child, DropFlagState::Present)
                });
            }
        }
    }
}

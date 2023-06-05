use crate::deref_separator::deref_finder;
use crate::MirPass;
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_mir_dataflow::elaborate_drops::{elaborate_drop, DropFlagState, Unwind};
use rustc_mir_dataflow::elaborate_drops::{DropElaborator, DropFlagMode, DropStyle};
use rustc_mir_dataflow::impls::{MaybeInitializedPlaces, MaybeUninitializedPlaces};
use rustc_mir_dataflow::move_paths::{LookupResult, MoveData, MovePathIndex};
use rustc_mir_dataflow::on_lookup_result_bits;
use rustc_mir_dataflow::un_derefer::UnDerefer;
use rustc_mir_dataflow::MoveDataParamEnv;
use rustc_mir_dataflow::{on_all_children_bits, on_all_drop_children_bits};
use rustc_mir_dataflow::{Analysis, ResultsCursor};
use rustc_span::Span;
use rustc_target::abi::{FieldIdx, VariantIdx};
use std::fmt;

/// During MIR building, Drop terminators are inserted in every place where a drop may occur.
/// However, in this phase, the presence of these terminators does not guarantee that a destructor will run,
/// as the target of the drop may be uninitialized.
/// In general, the compiler cannot determine at compile time whether a destructor will run or not.
///
/// At a high level, this pass refines Drop to only run the destructor if the
/// target is initialized. The way this is achieved is by inserting drop flags for every variable
/// that may be dropped, and then using those flags to determine whether a destructor should run.
/// Once this is complete, Drop terminators in the MIR correspond to a call to the "drop glue" or
/// "drop shim" for the type of the dropped place.
///
/// This pass relies on dropped places having an associated move path, which is then used to determine
/// the initialization status of the place and its descendants.
/// It's worth noting that a MIR containing a Drop without an associated move path is probably ill formed,
/// as it would allow running a destructor on a place behind a reference:
///
/// ```text
// fn drop_term<T>(t: &mut T) {
//     mir!(
//         {
//             Drop(*t, exit)
//         }
//         exit = {
//             Return()
//         }
//     )
// }
/// ```
pub struct ElaborateDrops;

impl<'tcx> MirPass<'tcx> for ElaborateDrops {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("elaborate_drops({:?} @ {:?})", body.source, body.span);

        let def_id = body.source.def_id();
        let param_env = tcx.param_env_reveal_all_normalized(def_id);
        let (side_table, move_data) = match MoveData::gather_moves(body, tcx, param_env) {
            Ok(move_data) => move_data,
            Err((move_data, _)) => {
                tcx.sess.delay_span_bug(
                    body.span,
                    "No `move_errors` should be allowed in MIR borrowck",
                );
                (Default::default(), move_data)
            }
        };
        let un_derefer = UnDerefer { tcx: tcx, derefer_sidetable: side_table };
        let elaborate_patch = {
            let env = MoveDataParamEnv { move_data, param_env };
            remove_dead_unwinds(tcx, body, &env, &un_derefer);

            let inits = MaybeInitializedPlaces::new(tcx, body, &env)
                .into_engine(tcx, body)
                .pass_name("elaborate_drops")
                .iterate_to_fixpoint()
                .into_results_cursor(body);

            let uninits = MaybeUninitializedPlaces::new(tcx, body, &env)
                .mark_inactive_variants_as_uninit()
                .into_engine(tcx, body)
                .pass_name("elaborate_drops")
                .iterate_to_fixpoint()
                .into_results_cursor(body);

            let reachable = traversal::reachable_as_bitset(body);

            let drop_flags = IndexVec::from_elem(None, &env.move_data.move_paths);
            ElaborateDropsCtxt {
                tcx,
                body,
                env: &env,
                init_data: InitializationData { inits, uninits },
                drop_flags,
                patch: MirPatch::new(body),
                un_derefer: un_derefer,
                reachable,
            }
            .elaborate()
        };
        elaborate_patch.apply(body);
        deref_finder(tcx, body);
    }
}

/// Removes unwind edges which are known to be unreachable, because they are in `drop` terminators
/// that can't drop anything.
fn remove_dead_unwinds<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    env: &MoveDataParamEnv<'tcx>,
    und: &UnDerefer<'tcx>,
) {
    debug!("remove_dead_unwinds({:?})", body.span);
    // We only need to do this pass once, because unwind edges can only
    // reach cleanup blocks, which can't have unwind edges themselves.
    let mut dead_unwinds = Vec::new();
    let mut flow_inits = MaybeInitializedPlaces::new(tcx, body, &env)
        .into_engine(tcx, body)
        .pass_name("remove_dead_unwinds")
        .iterate_to_fixpoint()
        .into_results_cursor(body);
    for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
        let place = match bb_data.terminator().kind {
            TerminatorKind::Drop { ref place, unwind: UnwindAction::Cleanup(_), .. } => {
                und.derefer(place.as_ref(), body).unwrap_or(*place)
            }
            _ => continue,
        };

        debug!("remove_dead_unwinds @ {:?}: {:?}", bb, bb_data);

        let LookupResult::Exact(path) = env.move_data.rev_lookup.find(place.as_ref()) else {
            debug!("remove_dead_unwinds: has parent; skipping");
            continue;
        };

        flow_inits.seek_before_primary_effect(body.terminator_loc(bb));
        debug!(
            "remove_dead_unwinds @ {:?}: path({:?})={:?}; init_data={:?}",
            bb,
            place,
            path,
            flow_inits.get()
        );

        let mut maybe_live = false;
        on_all_drop_children_bits(tcx, body, &env, path, |child| {
            maybe_live |= flow_inits.contains(child);
        });

        debug!("remove_dead_unwinds @ {:?}: maybe_live={}", bb, maybe_live);
        if !maybe_live {
            dead_unwinds.push(bb);
        }
    }

    if dead_unwinds.is_empty() {
        return;
    }

    let basic_blocks = body.basic_blocks.as_mut();
    for &bb in dead_unwinds.iter() {
        if let Some(unwind) = basic_blocks[bb].terminator_mut().unwind_mut() {
            *unwind = UnwindAction::Unreachable;
        }
    }
}

struct InitializationData<'mir, 'tcx> {
    inits: ResultsCursor<'mir, 'tcx, MaybeInitializedPlaces<'mir, 'tcx>>,
    uninits: ResultsCursor<'mir, 'tcx, MaybeUninitializedPlaces<'mir, 'tcx>>,
}

impl InitializationData<'_, '_> {
    fn seek_before(&mut self, loc: Location) {
        self.inits.seek_before_primary_effect(loc);
        self.uninits.seek_before_primary_effect(loc);
    }

    fn maybe_live_dead(&self, path: MovePathIndex) -> (bool, bool) {
        (self.inits.contains(path), self.uninits.contains(path))
    }
}

struct Elaborator<'a, 'b, 'tcx> {
    ctxt: &'a mut ElaborateDropsCtxt<'b, 'tcx>,
}

impl fmt::Debug for Elaborator<'_, '_, '_> {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<'a, 'tcx> DropElaborator<'a, 'tcx> for Elaborator<'a, '_, 'tcx> {
    type Path = MovePathIndex;

    fn patch(&mut self) -> &mut MirPatch<'tcx> {
        &mut self.ctxt.patch
    }

    fn body(&self) -> &'a Body<'tcx> {
        self.ctxt.body
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.ctxt.tcx
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.ctxt.param_env()
    }

    fn drop_style(&self, path: Self::Path, mode: DropFlagMode) -> DropStyle {
        let ((maybe_live, maybe_dead), multipart) = match mode {
            DropFlagMode::Shallow => (self.ctxt.init_data.maybe_live_dead(path), false),
            DropFlagMode::Deep => {
                let mut some_live = false;
                let mut some_dead = false;
                let mut children_count = 0;
                on_all_drop_children_bits(self.tcx(), self.body(), self.ctxt.env, path, |child| {
                    let (live, dead) = self.ctxt.init_data.maybe_live_dead(child);
                    debug!("elaborate_drop: state({:?}) = {:?}", child, (live, dead));
                    some_live |= live;
                    some_dead |= dead;
                    children_count += 1;
                });
                ((some_live, some_dead), children_count != 1)
            }
        };
        match (maybe_live, maybe_dead, multipart) {
            (false, _, _) => DropStyle::Dead,
            (true, false, _) => DropStyle::Static,
            (true, true, false) => DropStyle::Conditional,
            (true, true, true) => DropStyle::Open,
        }
    }

    fn clear_drop_flag(&mut self, loc: Location, path: Self::Path, mode: DropFlagMode) {
        match mode {
            DropFlagMode::Shallow => {
                self.ctxt.set_drop_flag(loc, path, DropFlagState::Absent);
            }
            DropFlagMode::Deep => {
                on_all_children_bits(
                    self.tcx(),
                    self.body(),
                    self.ctxt.move_data(),
                    path,
                    |child| self.ctxt.set_drop_flag(loc, child, DropFlagState::Absent),
                );
            }
        }
    }

    fn field_subpath(&self, path: Self::Path, field: FieldIdx) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.ctxt.move_data(), path, |e| match e {
            ProjectionElem::Field(idx, _) => idx == field,
            _ => false,
        })
    }

    fn array_subpath(&self, path: Self::Path, index: u64, size: u64) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.ctxt.move_data(), path, |e| match e {
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                debug_assert!(size == min_length, "min_length should be exact for arrays");
                assert!(!from_end, "from_end should not be used for array element ConstantIndex");
                offset == index
            }
            _ => false,
        })
    }

    fn deref_subpath(&self, path: Self::Path) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.ctxt.move_data(), path, |e| {
            e == ProjectionElem::Deref
        })
    }

    fn downcast_subpath(&self, path: Self::Path, variant: VariantIdx) -> Option<Self::Path> {
        rustc_mir_dataflow::move_path_children_matching(self.ctxt.move_data(), path, |e| match e {
            ProjectionElem::Downcast(_, idx) => idx == variant,
            _ => false,
        })
    }

    fn get_drop_flag(&mut self, path: Self::Path) -> Option<Operand<'tcx>> {
        self.ctxt.drop_flag(path).map(Operand::Copy)
    }
}

struct ElaborateDropsCtxt<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    env: &'a MoveDataParamEnv<'tcx>,
    init_data: InitializationData<'a, 'tcx>,
    drop_flags: IndexVec<MovePathIndex, Option<Local>>,
    patch: MirPatch<'tcx>,
    un_derefer: UnDerefer<'tcx>,
    reachable: BitSet<BasicBlock>,
}

impl<'b, 'tcx> ElaborateDropsCtxt<'b, 'tcx> {
    fn move_data(&self) -> &'b MoveData<'tcx> {
        &self.env.move_data
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.env.param_env
    }

    fn create_drop_flag(&mut self, index: MovePathIndex, span: Span) {
        let tcx = self.tcx;
        let patch = &mut self.patch;
        debug!("create_drop_flag({:?})", self.body.span);
        self.drop_flags[index].get_or_insert_with(|| patch.new_internal(tcx.types.bool, span));
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
            if !self.reachable.contains(bb) {
                continue;
            }
            let terminator = data.terminator();
            let place = match terminator.kind {
                TerminatorKind::Drop { ref place, .. } => {
                    self.un_derefer.derefer(place.as_ref(), self.body).unwrap_or(*place)
                }
                _ => continue,
            };

            self.init_data.seek_before(self.body.terminator_loc(bb));

            let path = self.move_data().rev_lookup.find(place.as_ref());
            debug!("collect_drop_flags: {:?}, place {:?} ({:?})", bb, place, path);

            let path = match path {
                LookupResult::Exact(e) => e,
                LookupResult::Parent(None) => continue,
                LookupResult::Parent(Some(parent)) => {
                    let (_maybe_live, maybe_dead) = self.init_data.maybe_live_dead(parent);

                    if self.body.local_decls[place.local].is_deref_temp() {
                        continue;
                    }

                    if maybe_dead {
                        self.tcx.sess.delay_span_bug(
                            terminator.source_info.span,
                            format!(
                                "drop of untracked, uninitialized value {:?}, place {:?} ({:?})",
                                bb, place, path
                            ),
                        );
                    }
                    continue;
                }
            };

            on_all_drop_children_bits(self.tcx, self.body, self.env, path, |child| {
                let (maybe_live, maybe_dead) = self.init_data.maybe_live_dead(child);
                debug!(
                    "collect_drop_flags: collecting {:?} from {:?}@{:?} - {:?}",
                    child,
                    place,
                    path,
                    (maybe_live, maybe_dead)
                );
                if maybe_live && maybe_dead {
                    self.create_drop_flag(child, terminator.source_info.span)
                }
            });
        }
    }

    fn elaborate_drops(&mut self) {
        for (bb, data) in self.body.basic_blocks.iter_enumerated() {
            if !self.reachable.contains(bb) {
                continue;
            }
            let loc = Location { block: bb, statement_index: data.statements.len() };
            let terminator = data.terminator();

            match terminator.kind {
                TerminatorKind::Drop { mut place, target, unwind, replace } => {
                    if let Some(new_place) = self.un_derefer.derefer(place.as_ref(), self.body) {
                        place = new_place;
                    }

                    self.init_data.seek_before(loc);
                    match self.move_data().rev_lookup.find(place.as_ref()) {
                        LookupResult::Exact(path) => {
                            let unwind = if data.is_cleanup {
                                Unwind::InCleanup
                            } else {
                                match unwind {
                                    UnwindAction::Cleanup(cleanup) => Unwind::To(cleanup),
                                    UnwindAction::Continue => Unwind::To(self.patch.resume_block()),
                                    UnwindAction::Unreachable => {
                                        Unwind::To(self.patch.unreachable_cleanup_block())
                                    }
                                    UnwindAction::Terminate => {
                                        Unwind::To(self.patch.terminate_block())
                                    }
                                }
                            };
                            elaborate_drop(
                                &mut Elaborator { ctxt: self },
                                terminator.source_info,
                                place,
                                path,
                                target,
                                unwind,
                                bb,
                            )
                        }
                        LookupResult::Parent(..) => {
                            if !replace {
                                self.tcx.sess.delay_span_bug(
                                    terminator.source_info.span,
                                    format!("drop of untracked value {:?}", bb),
                                );
                            }
                            // A drop and replace behind a pointer/array/whatever.
                            // The borrow checker requires that these locations are initialized before the assignment,
                            // so we just leave an unconditional drop.
                            assert!(!data.is_cleanup);
                        }
                    }
                }
                _ => continue,
            }
        }
    }

    fn constant_bool(&self, span: Span, val: bool) -> Rvalue<'tcx> {
        Rvalue::Use(Operand::Constant(Box::new(Constant {
            span,
            user_ty: None,
            literal: ConstantKind::from_bool(self.tcx, val),
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
            if !self.reachable.contains(bb) {
                continue;
            }
            if let TerminatorKind::Call {
                destination,
                target: Some(tgt),
                unwind: UnwindAction::Cleanup(_),
                ..
            } = data.terminator().kind
            {
                assert!(!self.patch.is_patched(bb));

                let loc = Location { block: tgt, statement_index: 0 };
                let path = self.move_data().rev_lookup.find(destination.as_ref());
                on_lookup_result_bits(self.tcx, self.body, self.move_data(), path, |child| {
                    self.set_drop_flag(loc, child, DropFlagState::Present)
                });
            }
        }
    }

    fn drop_flags_for_args(&mut self) {
        let loc = Location::START;
        rustc_mir_dataflow::drop_flag_effects_for_function_entry(
            self.tcx,
            self.body,
            self.env,
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
            if !self.reachable.contains(bb) {
                continue;
            }
            debug!("drop_flags_for_locs({:?})", data);
            for i in 0..(data.statements.len() + 1) {
                debug!("drop_flag_for_locs: stmt {}", i);
                if i == data.statements.len() {
                    match data.terminator().kind {
                        TerminatorKind::Drop { .. } => {
                            // drop elaboration should handle that by itself
                            continue;
                        }
                        TerminatorKind::Resume => {
                            // It is possible for `Resume` to be patched
                            // (in particular it can be patched to be replaced with
                            // a Goto; see `MirPatch::new`).
                        }
                        _ => {
                            assert!(!self.patch.is_patched(bb));
                        }
                    }
                }
                let loc = Location { block: bb, statement_index: i };
                rustc_mir_dataflow::drop_flag_effects_for_location(
                    self.tcx,
                    self.body,
                    self.env,
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
                unwind: UnwindAction::Continue | UnwindAction::Unreachable | UnwindAction::Terminate,
                ..
            } = data.terminator().kind
            {
                assert!(!self.patch.is_patched(bb));

                let loc = Location { block: bb, statement_index: data.statements.len() };
                let path = self.move_data().rev_lookup.find(destination.as_ref());
                on_lookup_result_bits(self.tcx, self.body, self.move_data(), path, |child| {
                    self.set_drop_flag(loc, child, DropFlagState::Present)
                });
            }
        }
    }
}

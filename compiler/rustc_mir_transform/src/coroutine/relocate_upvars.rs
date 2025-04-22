//! MIR rewrite pass to relocate upvars into native locals in the coroutine body
//!
//! # Summary
//! The current contract of coroutine upvars is as follows.
//! Coroutines are constructed, initially in state UNRESUMED, by copying or moving
//! captures into the `struct`-fields, which are also called prefix fields,
//! taking necessary references as per capture specification.
//!
//! ```text
//!  Low address                                                                                               High address
//!  ┌─────────┬─────────┬─────┬─────────────────────┬───────────────────────────────────────────────────────┬──────────────┐
//!  │         │         │     │                     │                                                       │              │
//!  │ Upvar 1 │ Upvar 2 │ ... │ Coroutine State Tag │ Ineligibles, aka. saved locals alive across 2+ states │ Other states │
//!  │         │         │     │                     │                                                       │              │
//!  └─────────┴─────────┴─────┴─────────────────────┴───────────────────────────────────────────────────────┴──────────────┘
//! ```
//!
//! In case some upvars are large and short-lived, the classic layout scheme can be wasteful.
//! One way to reduce the memory footprint is to
//!
//! This pass performs the following transformations.
//! 1. It generates a fresh batch of locals for each captured upvars.
//!
//! For each upvar, whether used or not, a fresh local is created with the same type.
//! The types respect the nature of the captures, being by-ref, by-ref-mut or by-value.
//! This is reflected in the results in the upvar analysis conducted in the HIR type-checking phase.
//!
//! 2. It replaces the places pointing into those upvars with places pointing into those locals instead
//!
//! Each place that starts with access into the coroutine structure `_1` is replaced with the fresh local as
//! the base.
//! Suppose we are to lower this coroutine into the MIR.
//! ```ignore (illustrative)
//! let mut captured = None;
//! let _ = #[coroutine] || {
//!     yield ();
//!     if let Some(inner) = &mut captured {
//!         *inner = 42i32; // (*)
//!     }
//! };
//! ```
//! `captured` is the only capture, whose mutable borrow is formally allotted to the first field `_1.0: &mut i32`.
//! The highlighted line `(*)` should be lowered, roughly, into MIR `(*_1.0) = const 42i32;`.
//! Now, by application of this pass, we create a new local `_4: &mut i32` and we perform the following
//! code transformation.
//!
//! A new block is constructed to just perform the relocation of this mutable borrow.
//! This block is inserted to the very beginning of the coroutine body control flow,
//! so that this is executed before any proper coroutine code as it transits from `UNRESUME` state to
//! any other state.
//! This "prologue" will look like the following.
//! ```ignore (illustrative)
//! StorageLive(_5);
//! StorageLive(_4);
//! _5 = move (_1.0);
//! _4 = move (_5);
//! StorageDead(_5);
//! ```
//! Note that we also create a trampoline local `_5` of the same type.
//!
//! ### Intricacy around the trampoline local
//! The reason that we need the trampolines is because state transformation and coroutine
//! layout calculation is not aware of potential storage conflict between captures as struct fields
//! and other saved locals.
//! The only guarantee that we can work with is one where any coroutine layout calculator respects
//! the storage conflict contracts between *MIR locals*.
//! It is known that calculators do not consider struct fields, where captures reside, as MIR locals.
//! This is the source of potential memory overlaps.
//! For instance, in a hypothetical situation,
//! - `_1.0` is relocated to `_4`;
//! - `_1.1` is relocated to `_6`;
//! - `_4` and `_6` remains live at one of the first suspension state;
//! - `_4` occupies the same offset of `_1.1` and `_6` occupies the same offset of `_1.0`
//!   as decided by some layout calculator;
//! In this scenario, without trampolining, the relocations introduce undefined behaviour.
//!
//! As a proposal for a future design, it is best that coroutine captures receive their own
//! MIR locals, possibly in a form of "function arguments" like `_1` itself.
//! The trampolining transformation already attempts to restore the semantics of MIR locals to
//! these captures and promoting them to "arguments" would make MIR safer to handle.
//!
//! One should note that this phase assumes that the initial built MIR respects the nature of captures.
//! For instance, if the upvar `_1.4` is instead a by-ref-mut capture of a value of type `T`,
//! this phase assumes that all access correctly built as operating on the place `(*_1.4)`.
//! Based on the assumption, this phase replaces `_1.4` with a fresh local `_34: &mut T` and
//! the correctness is still upheld.
//!
//! 3. It assembles an prologue to replace the current entry block.
//!
//! This prologue block transfers every captured upvar into its corresponding fresh local, *via scratch locals*.
//! The upvars are first completely moved into the scratch locals in batch, and then moved into the destination
//! locals in batch.
//! The reason is that it is possible that coroutine layout may change and the source memory location of
//! an upvar may not necessarily be mapped exactly to the same place as in the `UNRESUMED` state.
//! This is very possible, because the coroutine layout scheme at this moment remains opaque,
//! other than the contract that a saved local has a stable internal offset throughout its liveness span.
//!
//! While the current coroutine layout ensures that the same saved local has stable offsets throughout its lifetime,
//! technically the upvar in `UNRESUMED` state and their fresh locals are different saved locals.
//! This scratch locals re-establish safety so that the correct data permutation can take place,
//! when a future coroutine layout calculator sees the permutation fit.

use std::borrow::Cow;

use rustc_abi::FieldIdx;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::bug;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{
    self, BasicBlock, BasicBlockData, Body, Local, MirDumper, Place, ProjectionElem, START_BLOCK,
    SourceInfo, Statement, StatementKind, Terminator, TerminatorKind, UnwindAction,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::impls::{MaybeStorageLive, always_storage_live_locals};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol};
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use crate::pass_manager::MirPass;
use crate::patch::MirPatch;

pub(crate) struct RelocateUpvars(bool);

impl RelocateUpvars {
    pub(crate) fn new(do_relocate: bool) -> Self {
        Self(do_relocate)
    }
}

struct UpvarSubstitution<'tcx> {
    /// Newly minted local into which the upvar is moved
    local: Local,
    /// The temporary local that the prologue will permute the upvars with
    reloc: Local,
    /// Place into the capture structure where this upvar is found
    upvar_place: Place<'tcx>,
    /// The span of the captured upvar from the parent body
    span: Span,
    /// Name of the upvar
    name: Symbol,
}

struct SubstituteUpvarVisitor<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    mappings: &'a IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
}

impl<'tcx, 'a> MutVisitor<'tcx> for SubstituteUpvarVisitor<'tcx, 'a> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(
        &mut self,
        place: &mut Place<'tcx>,
        _context: mir::visit::PlaceContext,
        location: mir::Location,
    ) {
        if let Place { local: ty::CAPTURE_STRUCT_LOCAL, projection } = place
            && let [ProjectionElem::Field(field_idx, _ty), rest @ ..] = &***projection
        {
            let Some(&UpvarSubstitution { local, .. }) = self.mappings.get(*field_idx) else {
                bug!(
                    "SubstituteUpvar: found {field_idx:?} @ {location:?} but there is no upvar for it"
                )
            };
            let new_place = Place::from(local);
            let new_place = new_place.project_deeper(rest, self.tcx);
            *place = new_place;
        }
    }

    fn visit_terminator(
        &mut self,
        terminator: &mut mir::Terminator<'tcx>,
        location: mir::Location,
    ) {
        if let TerminatorKind::Drop { place, .. } = &terminator.kind
            && let Some(ty::CAPTURE_STRUCT_LOCAL) = place.as_local()
        {
            // This is a drop on the whole coroutine state, which we will processed later
            return;
        }
        self.super_terminator(terminator, location)
    }
}

#[derive(Debug, Clone, Copy)]
struct RelocationInfo {
    ident: Option<Ident>,
    immutable: bool,
    by_ref: bool,
}

impl RelocateUpvars {
    #[instrument(level = "debug", skip_all, fields(def_id = ?body.source))]
    pub(crate) fn run<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &mut Body<'tcx>,
        local_upvar_map: &mut IndexVec<FieldIdx, Option<Local>>,
    ) {
        if !self.0 {
            debug!("relocate upvar is set to no-op");
            return;
        }
        if body.yield_ty().is_none() {
            // It fails the litmus test as a coroutine
            debug!("not passing litmus test");
            return;
        }

        // The first argument is the coroutine type passed by value
        let coroutine_ty = if let Some(decl) = body.local_decls.get(ty::CAPTURE_STRUCT_LOCAL) {
            decl.ty
        } else {
            debug!("not coroutine ty, skipping");
            return;
        };

        // We only care when there is at least one upvar
        let (def_id, upvar_tys) = if let ty::Coroutine(def_id, args) = *coroutine_ty.kind() {
            let args = args.as_coroutine();
            (def_id, args.upvar_tys())
        } else {
            debug!("not coroutine ty again, skipping");
            return;
        };
        if upvar_tys.is_empty() {
            debug!("no upvar, skipping");
            return;
        }

        let upvar_infos = match body.source.instance {
            ty::InstanceKind::AsyncDropGlue(..) => {
                smallvec![RelocationInfo { ident: None, immutable: true, by_ref: false }]
            }
            ty::InstanceKind::Item(_) => tcx
                .closure_captures(def_id.expect_local())
                .iter()
                .map(|info| RelocationInfo {
                    ident: Some(Ident::new(info.to_symbol(), info.var_ident.span)),
                    immutable: matches!(info.mutability, ty::Mutability::Not),
                    by_ref: matches!(info.info.capture_kind, ty::UpvarCapture::ByRef(..)),
                })
                .collect::<SmallVec<[_; 4]>>(),
            ty::InstanceKind::Intrinsic(..)
            | ty::InstanceKind::VTableShim(..)
            | ty::InstanceKind::ReifyShim(..)
            | ty::InstanceKind::FnPtrShim(..)
            | ty::InstanceKind::Virtual(..)
            | ty::InstanceKind::ClosureOnceShim { .. }
            | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
            | ty::InstanceKind::ThreadLocalShim(..)
            | ty::InstanceKind::FutureDropPollShim(..)
            | ty::InstanceKind::DropGlue(..)
            | ty::InstanceKind::CloneShim(..)
            | ty::InstanceKind::AsyncDropGlueCtorShim(..)
            | ty::InstanceKind::FnPtrAddrShim(..) => unreachable!(),
        };

        if let Some(mir_dumper) = MirDumper::new(tcx, "RelocateUpvars", body) {
            mir_dumper.set_disambiguator(&"before").dump_mir(body);
        }

        let mut substitution_mapping = IndexVec::new();
        let mut patch = MirPatch::new(body);
        for (field_idx, (upvar_ty, &captured)) in upvar_tys.iter().zip(&upvar_infos).enumerate() {
            let span = captured.ident.map_or(DUMMY_SP, |ident| ident.span);
            let name = if let Some(ident) = captured.ident {
                ident.name
            } else {
                Symbol::intern(&format!("_{}", field_idx))
            };

            let immutable =
                if captured.immutable { mir::Mutability::Not } else { mir::Mutability::Mut };
            let local = patch.new_local_with_info(
                upvar_ty,
                span,
                mir::LocalInfo::User(mir::BindingForm::Var(mir::VarBindingForm {
                    binding_mode: rustc_ast::BindingMode(
                        if captured.by_ref {
                            rustc_ast::ByRef::Yes(immutable)
                        } else {
                            rustc_ast::ByRef::No
                        },
                        immutable,
                    ),
                    opt_ty_info: None,
                    opt_match_place: None,
                    pat_span: span,
                })),
                captured.immutable,
            );
            let reloc = patch.new_local_with_info(upvar_ty, span, mir::LocalInfo::Boring, true);

            let field_idx = FieldIdx::from_usize(field_idx);
            let upvar_place = Place::from(ty::CAPTURE_STRUCT_LOCAL)
                .project_deeper(&[ProjectionElem::Field(field_idx, upvar_ty)], tcx);

            substitution_mapping.push(UpvarSubstitution { local, reloc, upvar_place, span, name });
        }
        patch.apply(body);
        local_upvar_map.extend(substitution_mapping.iter().map(|sub| Some(sub.local)));
        SubstituteUpvarVisitor { tcx, mappings: &substitution_mapping }.visit_body(body);

        rewrite_drop_coroutine_struct(tcx, body, &substitution_mapping);
        insert_substitution_prologue(body, &substitution_mapping);
        patch_missing_storage_deads(tcx, body, &substitution_mapping);
        hydrate_var_debug_info(body, &substitution_mapping);
        if let Some(mir_dumper) = MirDumper::new(tcx, "RelocateUpvars", body) {
            mir_dumper.set_disambiguator(&"after").dump_mir(body);
        }
    }
}

impl<'tcx> MirPass<'tcx> for RelocateUpvars {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        self.run(tcx, body, &mut Default::default())
    }

    fn is_required(&self) -> bool {
        self.0
    }
}

fn rewrite_one_drop_coroutine_struct<'tcx>(
    tcx: TyCtxt<'tcx>,
    patch: &mut MirPatch<'tcx>,
    body: &Body<'tcx>,
    block: BasicBlock,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    let data = &body.basic_blocks[block];
    let source_info = data.terminator().source_info;
    let TerminatorKind::Drop {
        place: _,
        mut target,
        mut unwind,
        replace,
        drop: dropline,
        async_fut,
    } = data.terminator().kind
    else {
        unreachable!("unexpected terminator {:?}", data.terminator().kind)
    };
    let mut cleanup = match unwind {
        UnwindAction::Cleanup(tgt) => tgt,
        UnwindAction::Continue => patch.resume_block(),
        UnwindAction::Unreachable => patch.unreachable_cleanup_block(),
        UnwindAction::Terminate(reason) => patch.terminate_block(reason),
    };
    let typing_env = ty::TypingEnv::post_analysis(tcx, body.source.def_id());
    for &UpvarSubstitution { local, .. } in substitution_mapping {
        let place = local.into();
        let mut unwind_one = patch.new_block(BasicBlockData::new_stmts(
            vec![Statement::new(source_info, StatementKind::StorageDead(local))],
            Some(Terminator {
                source_info,
                kind: TerminatorKind::Goto {
                    target: if data.is_cleanup { target } else { cleanup },
                },
            }),
            true,
        ));
        let needs_drop = body.local_decls[local].ty.needs_drop(tcx, typing_env);
        let kind = if needs_drop {
            TerminatorKind::Drop {
                place,
                target: unwind_one,
                unwind: UnwindAction::Terminate(mir::UnwindTerminateReason::InCleanup),
                replace,
                drop: None,
                async_fut,
            }
        } else {
            TerminatorKind::Goto { target: unwind_one }
        };
        unwind_one = patch.new_block(BasicBlockData::new_stmts(
            vec![],
            Some(Terminator { source_info, kind }),
            true,
        ));
        if data.is_cleanup {
            unwind = UnwindAction::Cleanup(unwind_one);
            cleanup = unwind_one;
            target = unwind_one;
        } else {
            let mut drop_one = patch.new_block(BasicBlockData::new_stmts(
                vec![Statement::new(source_info, StatementKind::StorageDead(local))],
                Some(Terminator { source_info, kind: TerminatorKind::Goto { target } }),
                false,
            ));
            let kind = if needs_drop {
                TerminatorKind::Drop {
                    place,
                    target: drop_one,
                    unwind,
                    replace,
                    drop: dropline,
                    async_fut,
                }
            } else {
                TerminatorKind::Goto { target: drop_one }
            };
            drop_one = patch.new_block(BasicBlockData::new_stmts(
                vec![],
                Some(Terminator { source_info, kind }),
                false,
            ));
            target = drop_one;
            unwind = UnwindAction::Cleanup(unwind_one);
            cleanup = unwind_one;
        }
    }
    patch.patch_terminator(block, TerminatorKind::Goto { target });
}

fn rewrite_drop_coroutine_struct<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    let mut blocks = DenseBitSet::new_empty(body.basic_blocks.len());
    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let Terminator { source_info: _, kind: TerminatorKind::Drop { place, .. } } =
            block_data.terminator()
        else {
            continue;
        };
        let Some(local) = place.as_local() else { continue };
        if local == ty::CAPTURE_STRUCT_LOCAL {
            blocks.insert(block);
        }
    }
    let mut patch = MirPatch::new(body);
    for block in blocks.iter() {
        rewrite_one_drop_coroutine_struct(tcx, &mut patch, body, block, substitution_mapping);
    }
    patch.apply(body);
}

fn insert_substitution_prologue<'tcx>(
    body: &mut Body<'tcx>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    let mut patch = MirPatch::new(body);
    let mut stmts = Vec::with_capacity(2 * substitution_mapping.len());
    for &UpvarSubstitution { local, reloc, upvar_place, span, name: _ } in substitution_mapping {
        // For each upvar-local _$i
        let source_info = SourceInfo::outermost(span);
        // StorageLive(_$i)
        stmts.push(Statement::new(source_info, StatementKind::StorageLive(local)));
        // Use a fresh local _$i' here, so as to avoid potential field permutation
        // StorageLive(_$i')
        stmts.push(Statement::new(source_info, StatementKind::StorageLive(reloc)));
        // _$i' = move $<path>
        stmts.push(Statement::new(
            source_info,
            StatementKind::Assign(Box::new((
                reloc.into(),
                mir::Rvalue::Use(mir::Operand::Move(upvar_place)),
            ))),
        ));
    }
    for &UpvarSubstitution { local, reloc, upvar_place: _, span, name: _ } in substitution_mapping {
        let source_info = SourceInfo::outermost(span);
        // _$i = move $i'
        stmts.push(Statement::new(
            source_info,
            StatementKind::Assign(Box::new((
                local.into(),
                mir::Rvalue::Use(mir::Operand::Move(reloc.into())),
            ))),
        ));
        stmts.push(Statement::new(source_info, StatementKind::StorageDead(reloc)));
    }
    let source_info = SourceInfo::outermost(body.span);
    let prologue = patch.new_block(BasicBlockData::new_stmts(
        stmts,
        Some(Terminator { source_info, kind: TerminatorKind::Goto { target: START_BLOCK } }),
        false,
    ));
    patch.apply(body);

    // Manually patch so that prologue is the new entry-point
    let preds = body.basic_blocks.predecessors()[START_BLOCK].clone();
    let basic_blocks = body.basic_blocks.as_mut();
    for pred in preds {
        basic_blocks[pred].terminator_mut().successors_mut(|target| {
            if *target == START_BLOCK {
                *target = prologue;
            }
        });
    }
    basic_blocks.swap(START_BLOCK, prologue);
}

/// Occasionally there are upvar locals left without `StorageDead` because
/// they do not have destructors.
/// We need to mark them daed for correctness, as previously the entire
/// capture structure was marked dead and now we need to mark them one at
/// a time.
fn patch_missing_storage_deads<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    let always_live_locals = &always_storage_live_locals(body);

    let mut maybe_storage_live = MaybeStorageLive::new(Cow::Borrowed(always_live_locals))
        .iterate_to_fixpoint(tcx, body, None)
        .into_results_cursor(body);

    let mut upvar_locals = DenseBitSet::new_empty(body.local_decls.len());
    for subst in substitution_mapping {
        upvar_locals.insert(subst.local);
    }
    let mut storage_dead_stmts: IndexVec<BasicBlock, SmallVec<[_; 2]>> =
        IndexVec::from_elem_n(smallvec![], body.basic_blocks.len());
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if !data.is_cleanup && matches!(data.terminator().kind, TerminatorKind::Return) {
            let nr_stmts = data.statements.len();
            maybe_storage_live
                .seek_after_primary_effect(mir::Location { block, statement_index: nr_stmts });
            let mut missing_locals = maybe_storage_live.get().clone();
            missing_locals.intersect(&upvar_locals);
            let source_info = data.terminator().source_info;
            for local in missing_locals.iter() {
                storage_dead_stmts[block]
                    .push(mir::Statement::new(source_info, mir::StatementKind::StorageDead(local)));
            }
        }
    }
    let basic_blocks = body.basic_blocks.as_mut();
    for (block, storage_deaths) in storage_dead_stmts.iter_enumerated_mut() {
        basic_blocks[block].statements.extend(storage_deaths.drain(..));
    }
}

fn hydrate_var_debug_info<'tcx>(
    body: &mut Body<'tcx>,
    substitution_mapping: &IndexSlice<FieldIdx, UpvarSubstitution<'tcx>>,
) {
    for subst in substitution_mapping {
        body.var_debug_info.push(mir::VarDebugInfo {
            name: subst.name,
            source_info: SourceInfo::outermost(subst.span),
            composite: None,
            value: mir::VarDebugInfoContents::Place(Place::from(subst.local)),
            argument_index: None,
        });
    }
}

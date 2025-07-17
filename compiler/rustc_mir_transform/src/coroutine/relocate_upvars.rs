//! MIR rewrite pass to promote upvars into native locals in the coroutine body
//!
//! # Summary
//! This pass performs the following transformations.
//! 1. It generates a fresh batch of locals for each captured upvars.
//!
//! For each upvar, whether used or not, a fresh local is created with the same type.
//!
//! 2. It replaces the places pointing into those upvars with places pointing into those locals instead
//!
//! Each place that starts with access into the coroutine structure `_1` is replaced with the fresh local as
//! the base. For instance, `(_1.4 as Some).0` is rewritten into `(_34 as Some).0` when `_34` is the fresh local
//! corresponding to the captured upvar stored in `_1.4`.
//!
//! 3. It assembles an prologue to replace the current entry block.
//!
//! This prologue block transfers every captured upvar into its corresponding fresh local, *via scratch locals*.
//! The upvars are first completely moved into the scratch locals in batch, and then moved into the destination
//! locals in batch.
//! The reason is that it is possible that coroutine layout may change and the source memory location of
//! an upvar may not necessarily be mapped exactly to the same place as in the `Unresumed` state.
//! While coroutine layout ensures that the same saved local has stable offsets throughout its lifetime,
//! technically the upvar in `Unresumed` state and their fresh locals are different saved locals.
//! This scratch locals re-estabilish safety so that the correct data permutation can take place.

use std::borrow::Cow;

use rustc_abi::FieldIdx;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::bug;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{
    self, BasicBlock, BasicBlockData, Body, Local, Place, ProjectionElem, START_BLOCK, SourceInfo,
    Statement, StatementKind, Terminator, TerminatorKind, UnwindAction,
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

impl<'tcx> MirPass<'tcx> for RelocateUpvars {
    #[instrument(level = "debug", skip_all, fields(def_id = ?body.source))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
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
        body.local_upvar_map = substitution_mapping.iter().map(|sub| Some(sub.local)).collect();
        SubstituteUpvarVisitor { tcx, mappings: &substitution_mapping }.visit_body(body);

        rewrite_drop_coroutine_struct(body, &substitution_mapping);
        insert_substitution_prologue(body, &substitution_mapping);
        patch_missing_storage_deads(tcx, body, &substitution_mapping);
        hydrate_var_debug_info(body, &substitution_mapping);
    }

    fn is_required(&self) -> bool {
        true
    }
}

fn rewrite_one_drop_coroutine_struct<'tcx>(
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
        async_fut: None,
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
        unwind_one = patch.new_block(BasicBlockData::new_stmts(
            vec![],
            Some(Terminator {
                source_info,
                kind: TerminatorKind::Drop {
                    place,
                    target: unwind_one,
                    unwind: UnwindAction::Terminate(mir::UnwindTerminateReason::InCleanup),
                    replace,
                    drop: None,
                    async_fut: None,
                },
            }),
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
            drop_one = patch.new_block(BasicBlockData::new_stmts(
                vec![],
                Some(Terminator {
                    source_info,
                    kind: TerminatorKind::Drop {
                        place,
                        target: drop_one,
                        unwind,
                        replace,
                        drop: dropline,
                        async_fut: None,
                    },
                }),
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
        rewrite_one_drop_coroutine_struct(&mut patch, body, block, substitution_mapping);
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

    let mut patch = MirPatch::new(body);
    let mut upvar_locals = DenseBitSet::new_empty(body.local_decls.len());
    for subst in substitution_mapping {
        upvar_locals.insert(subst.local);
    }
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if !data.is_cleanup && matches!(data.terminator().kind, TerminatorKind::Return) {
            let nr_stmts = data.statements.len();
            maybe_storage_live
                .seek_after_primary_effect(mir::Location { block, statement_index: nr_stmts });
            let mut missing_locals = maybe_storage_live.get().clone();
            missing_locals.intersect(&upvar_locals);
            for (count, local) in missing_locals.iter().enumerate() {
                patch.add_statement(
                    mir::Location { block, statement_index: nr_stmts + count },
                    mir::StatementKind::StorageDead(local),
                );
            }
        }
    }
    patch.apply(body);
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

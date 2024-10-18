use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::bug;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Local, LocalInfo, Operand, Place, ProjectionElem, Rvalue,
    START_BLOCK, SourceInfo, Statement, StatementKind, Terminator, TerminatorKind, UnwindAction,
    UnwindTerminateReason,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::Span;
use rustc_target::abi::FieldIdx;

use crate::pass_manager::MirPass;

pub(crate) struct RelocateUpvars;

struct UpvarSubstitution<'tcx> {
    /// Newly minted local into which the upvar is moved
    local: Local,
    place: Place<'tcx>,
    /// Place into the capture structure where this upvar is found
    upvar_place: Place<'tcx>,
    span: Span,
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
        _context: rustc_middle::mir::visit::PlaceContext,
        location: rustc_middle::mir::Location,
    ) {
        if let Place { local: ty::CAPTURE_STRUCT_LOCAL, projection } = place
            && let [ProjectionElem::Field(field_idx, _ty), rest @ ..] = &***projection
        {
            let Some(&UpvarSubstitution { place: new_place, .. }) = self.mappings.get(*field_idx)
            else {
                bug!(
                    "SubstituteUpvar: found {field_idx:?} @ {location:?} but there is no upvar for it"
                )
            };
            let new_place = new_place.project_deeper(rest, self.tcx);
            *place = new_place;
        }
    }

    fn visit_terminator(
        &mut self,
        terminator: &mut rustc_middle::mir::Terminator<'tcx>,
        location: rustc_middle::mir::Location,
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

impl<'tcx> MirPass<'tcx> for RelocateUpvars {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let Some(coroutine_def_id) = body.source.def_id().as_local() else {
            return;
        };
        if tcx.coroutine_kind(coroutine_def_id).is_none() {
            return;
        }

        // The first argument is the coroutine type passed by value
        let coroutine_ty = if let Some(decl) = body.local_decls.get(ty::CAPTURE_STRUCT_LOCAL) {
            decl.ty
        } else {
            return;
        };

        // We only care when there is at least one upvar
        let (def_id, upvar_tys) = if let ty::Coroutine(def_id, args) = *coroutine_ty.kind() {
            let args = args.as_coroutine();
            (def_id, args.upvar_tys())
        } else {
            return;
        };
        if upvar_tys.is_empty() {
            return;
        }

        let upvar_infos = tcx.closure_captures(def_id.expect_local()).iter();

        let mut substitution_mapping = IndexVec::new();
        let mut patch = MirPatch::new(body);
        for (field_idx, (upvar_ty, &captured)) in upvar_tys.iter().zip(upvar_infos).enumerate() {
            let span = captured.var_ident.span;

            let local = patch.new_local_with_info(
                upvar_ty,
                span,
                LocalInfo::Boring,
                matches!(captured.mutability, ty::Mutability::Not),
            );

            let place = Place::from(local);
            let field_idx = FieldIdx::from_usize(field_idx);
            let upvar_place = Place::from(ty::CAPTURE_STRUCT_LOCAL)
                .project_deeper(&[ProjectionElem::Field(field_idx, upvar_ty)], tcx);

            substitution_mapping.push(UpvarSubstitution { local, place, upvar_place, span });
        }
        patch.apply(body);
        body.local_upvar_map = substitution_mapping.iter().map(|sub| Some(sub.local)).collect();
        SubstituteUpvarVisitor { tcx, mappings: &substitution_mapping }.visit_body(body);

        rewrite_drop_coroutine_struct(body, &substitution_mapping);
        insert_substitution_prologue(body, &substitution_mapping);
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
    let TerminatorKind::Drop { place: _, mut target, mut unwind, replace } = data.terminator().kind
    else {
        unreachable!()
    };
    let mut cleanup = match unwind {
        UnwindAction::Cleanup(tgt) => tgt,
        UnwindAction::Continue => patch.resume_block(),
        UnwindAction::Unreachable => patch.unreachable_cleanup_block(),
        UnwindAction::Terminate(reason) => patch.terminate_block(reason),
    };
    for &UpvarSubstitution { local, place, .. } in substitution_mapping {
        let mut unwind_one = patch.new_block(BasicBlockData {
            statements: vec![Statement { source_info, kind: StatementKind::StorageDead(local) }],
            terminator: Some(Terminator {
                source_info,
                kind: TerminatorKind::Goto {
                    target: if data.is_cleanup { target } else { cleanup },
                },
            }),
            is_cleanup: true,
        });
        unwind_one = patch.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info,
                kind: TerminatorKind::Drop {
                    place,
                    target: unwind_one,
                    unwind: UnwindAction::Terminate(UnwindTerminateReason::InCleanup),
                    replace,
                },
            }),
            is_cleanup: true,
        });
        if data.is_cleanup {
            unwind = UnwindAction::Cleanup(unwind_one);
            cleanup = unwind_one;
            target = unwind_one;
        } else {
            let mut drop_one = patch.new_block(BasicBlockData {
                statements: vec![Statement {
                    source_info,
                    kind: StatementKind::StorageDead(local),
                }],
                terminator: Some(Terminator { source_info, kind: TerminatorKind::Goto { target } }),
                is_cleanup: false,
            });
            drop_one = patch.new_block(BasicBlockData {
                terminator: Some(Terminator {
                    source_info,
                    kind: TerminatorKind::Drop { place, target: drop_one, unwind, replace },
                }),
                statements: vec![],
                is_cleanup: false,
            });
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
    let mut blocks = vec![];
    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        let Terminator { source_info: _, kind: TerminatorKind::Drop { place, .. } } =
            block_data.terminator()
        else {
            continue;
        };
        let Some(local) = place.as_local() else { continue };
        if local == ty::CAPTURE_STRUCT_LOCAL {
            blocks.push(block)
        }
    }
    let mut patch = MirPatch::new(body);
    for block in blocks {
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
    for &UpvarSubstitution { local, place, upvar_place, span } in substitution_mapping {
        // For each upvar-local _$i
        let source_info = SourceInfo::outermost(span);
        // StorageLive(_$i)
        stmts.push(Statement { source_info, kind: StatementKind::StorageLive(local) });
        // _$i = move $<path>
        stmts.push(Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((place, Rvalue::Use(Operand::Move(upvar_place))))),
        });
    }
    let source_info = SourceInfo::outermost(body.span);
    let resume_block = patch.resume_block();
    let prologue = patch.new_block(BasicBlockData {
        statements: stmts,
        terminator: Some(Terminator {
            source_info,
            kind: TerminatorKind::Drop {
                place: Place::from(ty::CAPTURE_STRUCT_LOCAL),
                target: START_BLOCK,
                unwind: UnwindAction::Cleanup(resume_block),
                replace: false,
            },
        }),
        is_cleanup: false,
    });
    patch.apply(body);

    // Manually patch so that prologue is the new entry
    let preds = body.basic_blocks.predecessors()[START_BLOCK].clone();
    let basic_blocks = body.basic_blocks.as_mut();
    for pred in preds {
        match &mut basic_blocks[pred].terminator_mut().kind {
            TerminatorKind::SwitchInt { discr: _, targets } => {
                for target in targets.all_targets_mut() {
                    if *target == START_BLOCK {
                        *target = prologue;
                    }
                }
            }
            TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::TailCall { .. } => unreachable!(),
            TerminatorKind::FalseUnwind { real_target: target, unwind: _ }
            | TerminatorKind::Goto { target }
            | TerminatorKind::Drop { place: _, target, unwind: _, replace: _ }
            | TerminatorKind::Assert { cond: _, expected: _, msg: _, target, unwind: _ }
            | TerminatorKind::Yield { value: _, resume: target, resume_arg: _, drop: _ } => {
                if *target == START_BLOCK {
                    *target = prologue;
                } else {
                    bug!()
                }
            }
            TerminatorKind::CoroutineDrop => {}
            TerminatorKind::Call {
                target,
                func: _,
                args: _,
                destination: _,
                unwind: _,
                call_source: _,
                fn_span: _,
            } => {
                if let Some(target) = target {
                    *target = prologue;
                } else {
                    bug!()
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                if *real_target == START_BLOCK {
                    *real_target = prologue;
                }
                if *imaginary_target == START_BLOCK {
                    *imaginary_target = prologue;
                }
            }
            TerminatorKind::InlineAsm {
                targets,
                template: _,
                operands: _,
                options: _,
                line_spans: _,
                unwind: _,
                asm_macro: _,
            } => {
                for target in targets {
                    if *target == START_BLOCK {
                        *target = prologue;
                    }
                }
            }
        }
    }
    basic_blocks.swap(START_BLOCK, prologue);
}

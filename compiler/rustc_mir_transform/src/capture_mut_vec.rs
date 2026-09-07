use rustc_data_structures::thin_vec::ThinVec;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_session::Session;
use rustc_span::sym;

use crate::{MirPass, PassPolicy};

/// Experimental copy-in/copy-out promotion for a narrowly constrained `&mut Vec<T>` argument.
///
/// This deliberately only accepts arguments whose uses are direct `Vec` method receiver operands,
/// including at least one `Vec::push` in a loop. Moving a `Vec` header changes its address, so
/// allowing arbitrary uses could make a raw pointer to that header observably different. The pass
/// runs after borrow checking and drop elaboration, and explicitly restores the header on normal
/// and unwind exits.
pub(super) struct CaptureMutVec;

impl<'tcx> MirPass<'tcx> for CaptureMutVec {
    fn policy(&self, sess: &Session) -> PassPolicy {
        PassPolicy::optimization(sess.mir_opt_level() >= 3)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let Some(vec_push) = tcx.get_diagnostic_item(sym::vec_push) else { return };
        let typing_env = body.typing_env(tcx);

        // Keep the first experiment simple: promote at most one concrete, small-element Vec.
        let candidate = (1..=body.arg_count).map(Local::from_usize).find(|&arg| {
            let ty::Ref(_, vec_ty, Mutability::Mut) = *body.local_decls[arg].ty.kind() else {
                return false;
            };
            let ty::Adt(def, args) = *vec_ty.kind() else { return false };
            if !tcx.is_diagnostic_item(sym::Vec, def.did()) {
                return false;
            }
            let elem = args.type_at(0);
            let Ok(layout) = tcx.layout_of(typing_env.as_query_input(elem)) else { return false };
            layout.size.bytes() <= 16 && only_used_by_vec_methods(tcx, body, arg, vec_push)
        });
        let Some(arg) = candidate else { return };

        let span = body.span;
        let source_info = SourceInfo::outermost(span);
        let ref_ty = body.local_decls[arg].ty;
        let ty::Ref(_, vec_ty, Mutability::Mut) = *ref_ty.kind() else { unreachable!() };
        let owned = body.local_decls.push(LocalDecl::new(vec_ty, span));
        let local_ref = body.local_decls.push(LocalDecl::new(ref_ty, span));
        let deref_arg = Place::from(arg).project_deeper(&[ProjectionElem::Deref], tcx);

        // Rewrite the receiver reborrows before adding the prologue, which itself uses `arg`.
        for data in body.basic_blocks.as_mut().iter_mut() {
            for statement in &mut data.statements {
                if let StatementKind::Assign(assign) = &mut statement.kind
                    && let Rvalue::Ref(_, _, place) = &mut assign.1
                    && place.local == arg
                {
                    place.local = local_ref;
                }
            }
        }

        let start = &mut body.basic_blocks.as_mut()[START_BLOCK].statements;
        start.insert(
            0,
            Statement::new(
                source_info,
                StatementKind::Assign(Box::new((
                    Place::from(owned),
                    Rvalue::Use(Operand::Move(deref_arg), WithRetag::Yes),
                ))),
            ),
        );
        start.insert(
            1,
            Statement::new(
                source_info,
                StatementKind::Assign(Box::new((
                    Place::from(local_ref),
                    Rvalue::Ref(
                        tcx.lifetimes.re_erased,
                        BorrowKind::Mut { kind: MutBorrowKind::Default },
                        Place::from(owned),
                    ),
                ))),
            ),
        );

        let restore = || {
            Statement::new(
                source_info,
                StatementKind::Assign(Box::new((
                    deref_arg,
                    Rvalue::Use(Operand::Move(Place::from(owned)), WithRetag::Yes),
                ))),
            )
        };

        let original_blocks = body.basic_blocks.len();
        // A single cleanup restores the header before propagating any unwind. Do not create an
        // unwind block for aborting targets, where all unwind actions have already been lowered.
        let needs_cleanup = body
            .basic_blocks
            .iter()
            .any(|data| matches!(data.terminator().unwind(), Some(UnwindAction::Continue)));
        let cleanup = needs_cleanup.then(|| {
            let mut cleanup_data = BasicBlockData::new(
                Some(Terminator {
                    source_info,
                    kind: TerminatorKind::UnwindResume,
                    attributes: ThinVec::new(),
                }),
                true,
            );
            cleanup_data.statements.push(restore());
            body.basic_blocks_mut().push(cleanup_data)
        });

        for data in body.basic_blocks.as_mut().iter_mut().take(original_blocks) {
            if matches!(
                data.terminator().kind,
                TerminatorKind::Return | TerminatorKind::UnwindResume
            ) {
                data.statements.push(restore());
            }
            if let Some(cleanup) = cleanup
                && let Some(unwind @ UnwindAction::Continue) = data.terminator_mut().unwind_mut()
            {
                *unwind = UnwindAction::Cleanup(cleanup);
            }
        }
    }
}

fn only_used_by_vec_methods(
    tcx: TyCtxt<'_>,
    body: &Body<'_>,
    arg: Local,
    vec_push: rustc_hir::def_id::DefId,
) -> bool {
    struct Uses {
        arg: Local,
        count: usize,
    }
    impl<'tcx> Visitor<'tcx> for Uses {
        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
            if place.local == self.arg && !matches!(context, PlaceContext::NonUse(_)) {
                self.count += 1;
            }
            self.super_place(place, context, location);
        }
    }

    let mut uses = Uses { arg, count: 0 };
    uses.visit_body(body);
    let mut receivers = Vec::new();
    for data in body.basic_blocks.iter() {
        for statement in &data.statements {
            if let StatementKind::Assign(assign) = &statement.kind
                && assign.0.projection.is_empty()
                && let Rvalue::Ref(_, _, place) = &assign.1
                && place.local == arg
            {
                receivers.push(assign.0.local);
            }
        }
    }
    let mut method_calls = 0;
    let mut push_in_loop = false;
    for (block, data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::Call { func, args, destination, .. } = &data.terminator().kind else {
            continue;
        };
        let Some(Operand::Move(receiver) | Operand::Copy(receiver)) =
            args.first().map(|arg| &arg.node)
        else {
            continue;
        };
        if !receiver.projection.is_empty() || !receivers.contains(&receiver.local) {
            continue;
        }
        let Some((did, _)) = func.const_fn_def() else { continue };
        let Some(item) = tcx.opt_associated_item(did) else { continue };
        if !item.is_method() || !matches!(item.container, ty::AssocContainer::InherentImpl) {
            continue;
        }
        // A method returning a value tied to the receiver may retain the address of the local
        // header (for example, `Vec::drain`). Proving that such a value does not escape requires
        // more dataflow than this initial pass performs.
        if body.local_decls[destination.local].ty.has_regions() {
            continue;
        }

        method_calls += 1;
        push_in_loop |= did == vec_push && block_is_in_cycle(body, block);
    }
    if !push_in_loop || uses.count != receivers.len() || method_calls != receivers.len() {
        return false;
    }

    // Each receiver temporary must occur exactly once as the assignment destination above and
    // exactly once as the receiver of a direct `Vec` method. In particular, reject any extra
    // escape of it.
    receivers.into_iter().all(|receiver| {
        let mut uses = Uses { arg: receiver, count: 0 };
        uses.visit_body(body);
        uses.count == 2
    })
}

fn block_is_in_cycle(body: &Body<'_>, start: BasicBlock) -> bool {
    let mut visited = vec![false; body.basic_blocks.len()];
    let mut pending: Vec<_> = body.basic_blocks[start].terminator().successors().collect();
    while let Some(block) = pending.pop() {
        if block == start {
            return true;
        }
        if !visited[block.index()] {
            visited[block.index()] = true;
            pending.extend(body.basic_blocks[block].terminator().successors());
        }
    }
    false
}

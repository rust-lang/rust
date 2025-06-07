//! This pass lowers calls to core::slice::len to just PtrMetadata op.
//! It should run before inlining!

use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{Ty, TyCtxt};

pub(super) struct LowerSliceLenCalls;

impl<'tcx> crate::MirPass<'tcx> for LowerSliceLenCalls {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // This pass also removes some otherwise-problematic retags, so we
        // always enable it when retags matter.
        sess.mir_opt_level() > 0 || sess.opts.unstable_opts.mir_emit_retag
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let language_items = tcx.lang_items();
        let Some(slice_len_fn_item_def_id) = language_items.slice_len_fn() else {
            // there is no lang item to compare to :)
            return;
        };

        // The one successor remains unchanged, so no need to invalidate
        let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
        for block in basic_blocks {
            // lower `<[_]>::len` calls
            lower_slice_len_call(tcx, &mut body.local_decls, block, slice_len_fn_item_def_id);
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

fn lower_slice_len_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
    block: &mut BasicBlockData<'tcx>,
    slice_len_fn_item_def_id: DefId,
) {
    let terminator = block.terminator();
    if let TerminatorKind::Call {
        func,
        args,
        destination,
        target: Some(bb),
        call_source: CallSource::Normal,
        ..
    } = &terminator.kind
        // some heuristics for fast rejection
        && let [arg] = &args[..]
        && let Some((fn_def_id, _)) = func.const_fn_def()
        && fn_def_id == slice_len_fn_item_def_id
    {
        // perform modifications from something like:
        //     _5 = core::slice::<impl [u8]>::len(move _6) -> bb1
        // into:
        //     _5 = PtrMetadata(move _6);
        //     goto bb1

        // make new RValue for Len
        let arg = arg.node.clone();
        let r_value = Rvalue::UnaryOp(UnOp::PtrMetadata, arg.clone());
        let len_statement_kind = StatementKind::Assign(Box::new((*destination, r_value)));
        let add_statement =
            Statement { kind: len_statement_kind, source_info: terminator.source_info };

        // modify terminator into simple Goto
        let new_terminator_kind = TerminatorKind::Goto { target: *bb };

        block.statements.push(add_statement);
        block.terminator_mut().kind = new_terminator_kind;

        if tcx.sess.opts.unstable_opts.mir_emit_retag {
            // If we care about retags, we want there to be *no retag* for this `len` call.
            // That means we have to clean up the MIR a little: the MIR will now often be:
            //     _6 = &*_4;
            //     _5 = PtrMetadata(move _6);
            // Change that to:
            //     _6 = &raw const *_4;
            //     _5 = PtrMetadata(move _6);
            let len = block.statements.len();
            if len >= 2
                && let &mut StatementKind::Assign(box (rebor_dest, ref mut rebor_r)) =
                    &mut block.statements[len - 2].kind
                && let Some(retag_local) = rebor_dest.as_local()
                && !local_decls[retag_local].is_user_variable()
                // The LHS of this previous assignment is the `arg` from above.
                && matches!(arg, Operand::Copy(p) | Operand::Move(p) if p.as_local() == Some(retag_local))
                // The RHS is a reference-taking operation.
                && let Rvalue::Ref(_, BorrowKind::Shared, rebor_orig) = *rebor_r
            {
                let orig_inner_ty = rebor_orig.ty(local_decls, tcx).ty;
                // Change the previous statement to use `&raw` instead of `&`.
                *rebor_r = Rvalue::RawPtr(RawPtrKind::FakeForPtrMetadata, rebor_orig);
                // Change the type of the local to match.
                local_decls[retag_local].ty = Ty::new_ptr(tcx, orig_inner_ty, Mutability::Not);
            }
            // There's a second pattern we need to recognize for `array.len()` calls. There,
            // the MIR is:
            //     _4 = &*_3;
            //     _6 = move _4 as &[T] (PointerCoercion(Unsize, Implicit));
            //     StorageDead(_4);
            //     _5 = PtrMetadata(move _6);
            // Change that to:
            //     _4 = &raw const *_3;
            //     _6 = move _4 as *const [T] (PointerCoercion(Unsize, Implicit));
            //     StorageDead(_4);
            //     _5 = PtrMetadata(move _6);
            if len >= 4
                && let [reborrow_stmt, unsize_stmt, storage_dead_stmt] =
                    block.statements.get_disjoint_mut([len - 4, len - 3, len - 2]).unwrap()
                // Check that the 3 statements have the right shape.
                && let &mut StatementKind::Assign(box (rebor_dest, ref mut rebor_r)) = &mut reborrow_stmt.kind
                && let Some(retag_local) = rebor_dest.as_local()
                && !local_decls[retag_local].is_user_variable()
                && let &mut StatementKind::Assign(box (unsized_dest, ref mut unsize_r)) = &mut unsize_stmt.kind
                && let Some(unsized_local) = unsized_dest.as_local()
                && !local_decls[unsized_local].is_user_variable()
                && storage_dead_stmt.kind == StatementKind::StorageDead(retag_local)
                // And check that they have the right operands.
                && matches!(arg, Operand::Copy(p) | Operand::Move(p) if p.as_local() == Some(unsized_local))
                && let &mut Rvalue::Cast(
                    CastKind::PointerCoercion(PointerCoercion::Unsize, CoercionSource::Implicit),
                    ref cast_op,
                    ref mut cast_ty,
                ) = unsize_r
                && let Operand::Copy(unsize_src) | Operand::Move(unsize_src) = cast_op
                && unsize_src.as_local() == Some(retag_local)
                && let Rvalue::Ref(_, BorrowKind::Shared, rebor_orig) = *rebor_r
            {
                let orig_inner_ty = rebor_orig.ty(local_decls, tcx).ty;
                let unsize_inner_ty = cast_ty.builtin_deref(true).unwrap();
                // Change the reborrow to use `&raw` instead of `&`.
                *rebor_r = Rvalue::RawPtr(RawPtrKind::FakeForPtrMetadata, rebor_orig);
                // Change the unsize coercion in the same vein.
                *cast_ty = Ty::new_ptr(tcx, unsize_inner_ty, Mutability::Not);
                // Change the type of the locals to match.
                local_decls[retag_local].ty = Ty::new_ptr(tcx, orig_inner_ty, Mutability::Not);
                local_decls[unsized_local].ty = *cast_ty;
            }
        }
    }
}

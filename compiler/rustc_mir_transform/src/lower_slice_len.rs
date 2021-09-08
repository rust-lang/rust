//! This pass lowers calls to core::slice::len to just Len op.
//! It should run before inlining!

use crate::MirPass;
use rustc_hir::def_id::DefId;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};

pub struct LowerSliceLenCalls;

impl<'tcx> MirPass<'tcx> for LowerSliceLenCalls {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        lower_slice_len_calls(tcx, body)
    }
}

pub fn lower_slice_len_calls<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let language_items = tcx.lang_items();
    let slice_len_fn_item_def_id = if let Some(slice_len_fn_item) = language_items.slice_len_fn() {
        slice_len_fn_item
    } else {
        // there is no language item to compare to :)
        return;
    };

    let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();

    for block in basic_blocks {
        // lower `<[_]>::len` calls
        lower_slice_len_call(tcx, block, &*local_decls, slice_len_fn_item_def_id);
    }
}

struct SliceLenPatchInformation<'tcx> {
    add_statement: Statement<'tcx>,
    new_terminator_kind: TerminatorKind<'tcx>,
}

fn lower_slice_len_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    block: &mut BasicBlockData<'tcx>,
    local_decls: &IndexVec<Local, LocalDecl<'tcx>>,
    slice_len_fn_item_def_id: DefId,
) {
    let mut patch_found: Option<SliceLenPatchInformation<'_>> = None;

    let terminator = block.terminator();
    match &terminator.kind {
        TerminatorKind::Call {
            func,
            args,
            destination: Some((dest, bb)),
            cleanup: None,
            from_hir_call: true,
            ..
        } => {
            // some heuristics for fast rejection
            if args.len() != 1 {
                return;
            }
            let arg = match args[0].place() {
                Some(arg) => arg,
                None => return,
            };
            let func_ty = func.ty(local_decls, tcx);
            match func_ty.kind() {
                ty::FnDef(fn_def_id, _) if fn_def_id == &slice_len_fn_item_def_id => {
                    // perform modifications
                    // from something like `_5 = core::slice::<impl [u8]>::len(move _6) -> bb1`
                    // into `_5 = Len(*_6)
                    // goto bb1

                    // make new RValue for Len
                    let deref_arg = tcx.mk_place_deref(arg);
                    let r_value = Rvalue::Len(deref_arg);
                    let len_statement_kind = StatementKind::Assign(Box::new((*dest, r_value)));
                    let add_statement = Statement {
                        kind: len_statement_kind,
                        source_info: terminator.source_info.clone(),
                    };

                    // modify terminator into simple Goto
                    let new_terminator_kind = TerminatorKind::Goto { target: bb.clone() };

                    let patch = SliceLenPatchInformation { add_statement, new_terminator_kind };

                    patch_found = Some(patch);
                }
                _ => {}
            }
        }
        _ => {}
    }

    if let Some(SliceLenPatchInformation { add_statement, new_terminator_kind }) = patch_found {
        block.statements.push(add_statement);
        block.terminator_mut().kind = new_terminator_kind;
    }
}

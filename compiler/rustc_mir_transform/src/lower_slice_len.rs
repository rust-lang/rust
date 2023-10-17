//! This pass lowers calls to core::slice::len to just Len op.
//! It should run before inlining!

use crate::MirPass;
use rustc_hir::def_id::DefId;
use rustc_index::IndexSlice;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};

pub struct LowerSliceLenCalls;

impl<'tcx> MirPass<'tcx> for LowerSliceLenCalls {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        lower_slice_len_calls(tcx, body)
    }
}

pub fn lower_slice_len_calls<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let language_items = tcx.lang_items();
    let Some(slice_len_fn_item_def_id) = language_items.slice_len_fn() else {
        // there is no language item to compare to :)
        return;
    };

    // The one successor remains unchanged, so no need to invalidate
    let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
    for block in basic_blocks {
        // lower `<[_]>::len` calls
        lower_slice_len_call(tcx, block, &body.local_decls, slice_len_fn_item_def_id);
    }
}

fn lower_slice_len_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    block: &mut BasicBlockData<'tcx>,
    local_decls: &IndexSlice<Local, LocalDecl<'tcx>>,
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
        && let Some(arg) = arg.place()
        && let ty::FnDef(fn_def_id, _) = func.ty(local_decls, tcx).kind()
        && *fn_def_id == slice_len_fn_item_def_id
    {
        // perform modifications from something like:
        //     _5 = core::slice::<impl [u8]>::len(move _6) -> bb1
        // into:
        //     _5 = Len(*_6)
        //     goto bb1

        // make new RValue for Len
        let deref_arg = tcx.mk_place_deref(arg);
        let r_value = Rvalue::Len(deref_arg);
        let len_statement_kind = StatementKind::Assign(Box::new((*destination, r_value)));
        let add_statement =
            Statement { kind: len_statement_kind, source_info: terminator.source_info };

        // modify terminator into simple Goto
        let new_terminator_kind = TerminatorKind::Goto { target: *bb };

        block.statements.push(add_statement);
        block.terminator_mut().kind = new_terminator_kind;
    }
}

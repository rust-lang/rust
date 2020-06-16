use crate::transform::{MirPass, MirSource};
use crate::util::patch::MirPatch;
use rustc_hir::lang_items;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::Span;

/// Inserts call to count_code_region() as a placeholder to be replaced during code generation with
/// the intrinsic llvm.instrprof.increment.
pub struct InstrumentCoverage;

impl<'tcx> MirPass<'tcx> for InstrumentCoverage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.instrument_coverage {
            debug!("instrumenting {:?}", src.def_id());
            instrument_coverage(tcx, body);
        }
    }
}

// The first counter (start of the function) is index zero.
const INIT_FUNCTION_COUNTER: u32 = 0;

/// Injects calls to placeholder function `count_code_region()`.
// FIXME(richkadel): As a first step, counters are only injected at the top of each function.
// The complete solution will inject counters at each conditional code branch.
pub fn instrument_coverage<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let span = body.span.shrink_to_lo();

    let count_code_region_fn = function_handle(
        tcx,
        tcx.require_lang_item(lang_items::CountCodeRegionFnLangItem, None),
        span,
    );
    let counter_index = Operand::const_from_scalar(
        tcx,
        tcx.types.u32,
        Scalar::from_u32(INIT_FUNCTION_COUNTER),
        span,
    );

    let mut patch = MirPatch::new(body);

    let new_block = patch.new_block(placeholder_block(SourceInfo::outermost(body.span)));
    let next_block = START_BLOCK;

    let temp = patch.new_temp(tcx.mk_unit(), body.span);
    patch.patch_terminator(
        new_block,
        TerminatorKind::Call {
            func: count_code_region_fn,
            args: vec![counter_index],
            // new_block will swapped with the next_block, after applying patch
            destination: Some((Place::from(temp), new_block)),
            cleanup: None,
            from_hir_call: false,
            fn_span: span,
        },
    );

    patch.add_statement(new_block.start_location(), StatementKind::StorageLive(temp));
    patch.add_statement(next_block.start_location(), StatementKind::StorageDead(temp));

    patch.apply(body);

    // To insert the `new_block` in front of the first block in the counted branch (for example,
    // the START_BLOCK, at the top of the function), just swap the indexes, leaving the rest of the
    // graph unchanged.
    body.basic_blocks_mut().swap(next_block, new_block);
}

fn function_handle<'tcx>(tcx: TyCtxt<'tcx>, fn_def_id: DefId, span: Span) -> Operand<'tcx> {
    let ret_ty = tcx.fn_sig(fn_def_id).output();
    let ret_ty = ret_ty.no_bound_vars().unwrap();
    let substs = tcx.mk_substs(::std::iter::once(ty::subst::GenericArg::from(ret_ty)));
    Operand::function_handle(tcx, fn_def_id, substs, span)
}

fn placeholder_block<'tcx>(source_info: SourceInfo) -> BasicBlockData<'tcx> {
    BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info,
            // this gets overwritten by the counter Call
            kind: TerminatorKind::Unreachable,
        }),
        is_cleanup: false,
    }
}

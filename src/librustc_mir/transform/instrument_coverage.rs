use crate::transform::{MirPass, MirSource};
use rustc_index::vec::Idx;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::mir::{Local, LocalDecl};
use rustc_middle::ty;
use rustc_middle::ty::Ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::Span;

pub struct InstrumentCoverage;

/**
 * Inserts call to count_code_region() as a placeholder to be replaced during code generation with
 * the intrinsic llvm.instrprof.increment.
 */

// FIXME(richkadel): As a first step, counters are only injected at the top of each function.
// The complete solution will inject counters at each conditional code branch.

impl<'tcx> MirPass<'tcx> for InstrumentCoverage {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.instrument_coverage {
            if let Some(callee_fn_def_id) = tcx.lang_items().count_code_region_fn() {
                debug!("instrumenting {:?}", src.def_id());
                instrument_coverage(tcx, callee_fn_def_id, body);
            }
        }
    }
}

pub fn instrument_coverage<'tcx>(
    tcx: TyCtxt<'tcx>,
    callee_fn_def_id: DefId,
    body: &mut Body<'tcx>,
) {
    let span = body.span.shrink_to_lo();

    let ret_ty = tcx.fn_sig(callee_fn_def_id).output();
    let ret_ty = ret_ty.no_bound_vars().unwrap();
    let substs = tcx.mk_substs(::std::iter::once(ty::subst::GenericArg::from(ret_ty)));

    let count_code_region_fn: Operand<'_> =
        Operand::function_handle(tcx, callee_fn_def_id, substs, span);

    let index = const_int_operand(tcx, span.clone(), tcx.types.u32, 0);

    let args = vec![index];

    let source_info = SourceInfo { span: span, scope: OUTERMOST_SOURCE_SCOPE };

    let new_block = START_BLOCK + body.basic_blocks().len();

    let next_local = body.local_decls.len();
    let new_temp = Local::new(next_local);
    let unit_temp = Place::from(new_temp);

    let storage_live = Statement { source_info, kind: StatementKind::StorageLive(new_temp) };
    let storage_dead = Statement { source_info, kind: StatementKind::StorageDead(new_temp) };

    let count_code_region_call = TerminatorKind::Call {
        func: count_code_region_fn,
        args,
        destination: Some((unit_temp, new_block)),
        cleanup: None,
        from_hir_call: false,
    };

    body.local_decls.push(LocalDecl::new(tcx.mk_unit(), body.span));
    body.basic_blocks_mut().push(BasicBlockData {
        statements: vec![storage_live],
        is_cleanup: false,
        terminator: Some(Terminator { source_info, kind: count_code_region_call }),
    });

    body.basic_blocks_mut().swap(START_BLOCK, new_block);
    body[new_block].statements.push(storage_dead);

    // FIXME(richkadel): ALSO add each computed Span for each conditional branch to the coverage map
    // and provide that map to LLVM to encode in the final binary.
}

fn const_int_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    ty: Ty<'tcx>,
    val: u128,
) -> Operand<'tcx> {
    let param_env_and_ty = ty::ParamEnv::empty().and(ty);
    let size = tcx
        .layout_of(param_env_and_ty)
        .unwrap_or_else(|e| panic!("could not compute layout for {:?}: {:?}", ty, e))
        .size;
    Operand::Constant(box Constant {
        span,
        user_ty: None,
        literal: ty::Const::from_scalar(tcx, Scalar::from_uint(val, size), ty),
    })
}

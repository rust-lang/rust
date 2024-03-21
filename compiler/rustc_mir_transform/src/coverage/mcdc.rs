use rustc_macros::Diagnostic;
use rustc_middle::{
    mir::{
        self,
        coverage::{CoverageKind, DecisionSpan},
        Statement,
    },
    ty::TyCtxt,
};
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(mir_transform_mcdc_too_many_conditions)]
pub(crate) struct MCDCTooManyConditions {
    #[primary_span]
    pub span: Span,
    pub num_conditions: u32,
    pub max_conditions: u32,
}

const MAX_COND_DECISION: u32 = 6;

/// If MCDC coverage is enabled, add MCDC instrumentation to the function.
/// Will do the following things :
/// 1. Add an instruction in the first basic block for the codegen to call
///     the `instrprof.mcdc.parameters` instrinsic.
pub fn instrument_function_mcdc<'tcx>(tcx: TyCtxt<'tcx>, mir_body: &mut mir::Body<'tcx>) {
    if !tcx.sess.instrument_coverage_mcdc() {
        return;
    }

    let Some(branch_info) = mir_body.coverage_branch_info.as_deref() else {
        return;
    };

    // Compute the total sum of needed bytes for the current body.
    let needed_bytes: u32 = branch_info
        .decision_spans
        .iter()
        .filter_map(|DecisionSpan { span, num_conditions }| {
            if *num_conditions > MAX_COND_DECISION {
                tcx.dcx().emit_warn(MCDCTooManyConditions {
                    span: *span,
                    num_conditions: *num_conditions,
                    max_conditions: MAX_COND_DECISION,
                });
                None
            } else {
                Some(1 << num_conditions)
            }
        })
        .sum();

    let entry_bb = &mut mir_body.basic_blocks_mut()[mir::START_BLOCK];
    let mcdc_parameters_statement = Statement {
        source_info: entry_bb.terminator().source_info,
        kind: mir::StatementKind::Coverage(CoverageKind::MCDCBitmapRequire { needed_bytes }),
    };
    entry_bb.statements.insert(0, mcdc_parameters_statement);
}

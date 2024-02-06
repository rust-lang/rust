use rustc_errors::{AddToDiagnostic, DiagnosticBuilder, EmissionGuarantee, SubdiagnosticMessageOp};
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::thir::Pat;
use rustc_middle::ty::Ty;
use rustc_span::Span;

use crate::rustc::{RustcMatchCheckCtxt, WitnessPat};

#[derive(Subdiagnostic)]
#[label(pattern_analysis_uncovered)]
pub struct Uncovered<'tcx> {
    #[primary_span]
    span: Span,
    count: usize,
    witness_1: Pat<'tcx>,
    witness_2: Pat<'tcx>,
    witness_3: Pat<'tcx>,
    remainder: usize,
}

impl<'tcx> Uncovered<'tcx> {
    pub fn new<'p>(
        span: Span,
        cx: &RustcMatchCheckCtxt<'p, 'tcx>,
        witnesses: Vec<WitnessPat<'p, 'tcx>>,
    ) -> Self
    where
        'tcx: 'p,
    {
        let witness_1 = cx.hoist_witness_pat(witnesses.get(0).unwrap());
        Self {
            span,
            count: witnesses.len(),
            // Substitute dummy values if witnesses is smaller than 3. These will never be read.
            witness_2: witnesses
                .get(1)
                .map(|w| cx.hoist_witness_pat(w))
                .unwrap_or_else(|| witness_1.clone()),
            witness_3: witnesses
                .get(2)
                .map(|w| cx.hoist_witness_pat(w))
                .unwrap_or_else(|| witness_1.clone()),
            witness_1,
            remainder: witnesses.len().saturating_sub(3),
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(pattern_analysis_overlapping_range_endpoints)]
#[note]
pub struct OverlappingRangeEndpoints<'tcx> {
    #[label]
    pub range: Span,
    #[subdiagnostic]
    pub overlap: Vec<Overlap<'tcx>>,
}

pub struct Overlap<'tcx> {
    pub span: Span,
    pub range: Pat<'tcx>,
}

impl<'tcx> AddToDiagnostic for Overlap<'tcx> {
    fn add_to_diagnostic_with<G: EmissionGuarantee, F: SubdiagnosticMessageOp<G>>(
        self,
        diag: &mut DiagnosticBuilder<'_, G>,
        _: F,
    ) {
        let Overlap { span, range } = self;

        // FIXME(mejrs) unfortunately `#[derive(LintDiagnostic)]`
        // does not support `#[subdiagnostic(eager)]`...
        let message = format!("this range overlaps on `{range}`...");
        diag.span_label(span, message);
    }
}

#[derive(LintDiagnostic)]
#[diag(pattern_analysis_non_exhaustive_omitted_pattern)]
#[help]
#[note]
pub(crate) struct NonExhaustiveOmittedPattern<'tcx> {
    pub scrut_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub uncovered: Uncovered<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(pattern_analysis_non_exhaustive_omitted_pattern_lint_on_arm)]
#[help]
pub(crate) struct NonExhaustiveOmittedPatternLintOnArm {
    #[label]
    pub lint_span: Span,
    #[suggestion(code = "#[{lint_level}({lint_name})]\n", applicability = "maybe-incorrect")]
    pub suggest_lint_on_match: Option<Span>,
    pub lint_level: &'static str,
    pub lint_name: &'static str,
}

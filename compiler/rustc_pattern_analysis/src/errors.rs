use rustc_errors::{AddToDiagnostic, Diag, EmissionGuarantee, SubdiagMessageOp};
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
    fn add_to_diagnostic_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
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
#[diag(pattern_analysis_excluside_range_missing_max)]
pub struct ExclusiveRangeMissingMax<'tcx> {
    #[label]
    #[suggestion(code = "{suggestion}", applicability = "maybe-incorrect")]
    /// This is an exclusive range that looks like `lo..max` (i.e. doesn't match `max`).
    pub first_range: Span,
    /// Suggest `lo..=max` instead.
    pub suggestion: String,
    pub max: Pat<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(pattern_analysis_excluside_range_missing_gap)]
pub struct ExclusiveRangeMissingGap<'tcx> {
    #[label]
    #[suggestion(code = "{suggestion}", applicability = "maybe-incorrect")]
    /// This is an exclusive range that looks like `lo..gap` (i.e. doesn't match `gap`).
    pub first_range: Span,
    pub gap: Pat<'tcx>,
    /// Suggest `lo..=gap` instead.
    pub suggestion: String,
    #[subdiagnostic]
    /// All these ranges skipped over `gap` which we think is probably a mistake.
    pub gap_with: Vec<GappedRange<'tcx>>,
}

pub struct GappedRange<'tcx> {
    pub span: Span,
    pub gap: Pat<'tcx>,
    pub first_range: Pat<'tcx>,
}

impl<'tcx> AddToDiagnostic for GappedRange<'tcx> {
    fn add_to_diagnostic_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _: F,
    ) {
        let GappedRange { span, gap, first_range } = self;

        // FIXME(mejrs) unfortunately `#[derive(LintDiagnostic)]`
        // does not support `#[subdiagnostic(eager)]`...
        let message = format!(
            "this could appear to continue range `{first_range}`, but `{gap}` isn't matched by \
            either of them"
        );
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

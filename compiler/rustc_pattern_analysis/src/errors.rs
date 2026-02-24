use rustc_errors::{Diag, EmissionGuarantee, Subdiagnostic};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::Span;

use crate::rustc::{RustcPatCtxt, WitnessPat};

#[derive(Subdiagnostic)]
#[label(
    "{$count ->
        [1] pattern `{$witness_1}`
        [2] patterns `{$witness_1}` and `{$witness_2}`
        [3] patterns `{$witness_1}`, `{$witness_2}` and `{$witness_3}`
        *[other] patterns `{$witness_1}`, `{$witness_2}`, `{$witness_3}` and {$remainder} more
    } not covered"
)]
pub struct Uncovered {
    #[primary_span]
    span: Span,
    count: usize,
    witness_1: String, // a printed pattern
    witness_2: String, // a printed pattern
    witness_3: String, // a printed pattern
    remainder: usize,
}

impl Uncovered {
    pub fn new<'p, 'tcx>(
        span: Span,
        cx: &RustcPatCtxt<'p, 'tcx>,
        witnesses: Vec<WitnessPat<'p, 'tcx>>,
    ) -> Self
    where
        'tcx: 'p,
    {
        let witness_1 = cx.print_witness_pat(witnesses.get(0).unwrap());
        Self {
            span,
            count: witnesses.len(),
            // Substitute dummy values if witnesses is smaller than 3. These will never be read.
            witness_2: witnesses.get(1).map(|w| cx.print_witness_pat(w)).unwrap_or_default(),
            witness_3: witnesses.get(2).map(|w| cx.print_witness_pat(w)).unwrap_or_default(),
            witness_1,
            remainder: witnesses.len().saturating_sub(3),
        }
    }
}

#[derive(Diagnostic)]
#[diag("multiple patterns overlap on their endpoints")]
#[note("you likely meant to write mutually exclusive ranges")]
pub struct OverlappingRangeEndpoints {
    #[label("... with this range")]
    pub range: Span,
    #[subdiagnostic]
    pub overlap: Vec<Overlap>,
}

#[derive(Subdiagnostic)]
#[label("this range overlaps on `{$range}`...")]
pub struct Overlap {
    #[primary_span]
    pub span: Span,
    pub range: String, // a printed pattern
}

#[derive(Diagnostic)]
#[diag("exclusive range missing `{$max}`")]
pub struct ExclusiveRangeMissingMax {
    #[label("this range doesn't match `{$max}` because `..` is an exclusive range")]
    #[suggestion(
        "use an inclusive range instead",
        code = "{suggestion}",
        applicability = "maybe-incorrect"
    )]
    /// This is an exclusive range that looks like `lo..max` (i.e. doesn't match `max`).
    pub first_range: Span,
    /// Suggest `lo..=max` instead.
    pub suggestion: String,
    pub max: String, // a printed pattern
}

#[derive(Diagnostic)]
#[diag("multiple ranges are one apart")]
pub struct ExclusiveRangeMissingGap {
    #[label("this range doesn't match `{$gap}` because `..` is an exclusive range")]
    #[suggestion(
        "use an inclusive range instead",
        code = "{suggestion}",
        applicability = "maybe-incorrect"
    )]
    /// This is an exclusive range that looks like `lo..gap` (i.e. doesn't match `gap`).
    pub first_range: Span,
    pub gap: String, // a printed pattern
    /// Suggest `lo..=gap` instead.
    pub suggestion: String,
    #[subdiagnostic]
    /// All these ranges skipped over `gap` which we think is probably a mistake.
    pub gap_with: Vec<GappedRange>,
}

pub struct GappedRange {
    pub span: Span,
    pub gap: String,         // a printed pattern
    pub first_range: String, // a printed pattern
}

impl Subdiagnostic for GappedRange {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
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

#[derive(Diagnostic)]
#[diag("some variants are not matched explicitly")]
#[help("ensure that all variants are matched explicitly by adding the suggested match arms")]
#[note(
    "the matched value is of type `{$scrut_ty}` and the `non_exhaustive_omitted_patterns` attribute was found"
)]
pub(crate) struct NonExhaustiveOmittedPattern<'tcx> {
    pub scrut_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub uncovered: Uncovered,
}

#[derive(LintDiagnostic)]
#[diag("the lint level must be set on the whole match")]
#[help("it no longer has any effect to set the lint level on an individual match arm")]
pub(crate) struct NonExhaustiveOmittedPatternLintOnArm {
    #[label("remove this attribute")]
    pub lint_span: Span,
    #[suggestion(
        "set the lint level on the whole match",
        code = "#[{lint_level}({lint_name})]\n",
        applicability = "maybe-incorrect"
    )]
    pub suggest_lint_on_match: Option<Span>,
    pub lint_level: &'static str,
    pub lint_name: &'static str,
}

#[derive(Diagnostic)]
#[diag("mix of deref patterns and normal constructors")]
pub(crate) struct MixedDerefPatternConstructors<'tcx> {
    #[primary_span]
    pub spans: Vec<Span>,
    pub smart_pointer_ty: Ty<'tcx>,
    #[label("matches on the result of dereferencing `{$smart_pointer_ty}`")]
    pub deref_pattern_label: Span,
    #[label("matches directly on `{$smart_pointer_ty}`")]
    pub normal_constructor_label: Span,
}

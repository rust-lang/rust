use rustc_middle::lint::LevelAndSource;
use rustc_session::lint::builtin::NON_EXHAUSTIVE_OMITTED_PATTERNS;
use rustc_span::ErrorGuaranteed;
use tracing::instrument;

use crate::MatchArm;
use crate::constructor::Constructor;
use crate::errors::{NonExhaustiveOmittedPattern, NonExhaustiveOmittedPatternLintOnArm, Uncovered};
use crate::pat_column::PatternColumn;
use crate::rustc::{RevealedTy, RustcPatCtxt, WitnessPat};

/// Traverse the patterns to collect any variants of a non_exhaustive enum that fail to be mentioned
/// in a given column.
#[instrument(level = "debug", skip(cx), ret)]
fn collect_nonexhaustive_missing_variants<'p, 'tcx>(
    cx: &RustcPatCtxt<'p, 'tcx>,
    column: &PatternColumn<'p, RustcPatCtxt<'p, 'tcx>>,
) -> Result<Vec<WitnessPat<'p, 'tcx>>, ErrorGuaranteed> {
    let Some(&ty) = column.head_ty() else {
        return Ok(Vec::new());
    };

    let set = column.analyze_ctors(cx, &ty)?;
    if set.present.is_empty() {
        // We can't consistently handle the case where no constructors are present (since this would
        // require digging deep through any type in case there's a non_exhaustive enum somewhere),
        // so for consistency we refuse to handle the top-level case, where we could handle it.
        return Ok(Vec::new());
    }

    let mut witnesses = Vec::new();
    if cx.is_foreign_non_exhaustive_enum(ty) {
        witnesses.extend(
            set.missing
                .into_iter()
                // This will list missing visible variants.
                .filter(|c| !matches!(c, Constructor::Hidden | Constructor::NonExhaustive))
                .map(|missing_ctor| WitnessPat::wild_from_ctor(cx, missing_ctor, ty)),
        )
    }

    // Recurse into the fields.
    for ctor in set.present {
        let specialized_columns = column.specialize(cx, &ty, &ctor);
        let wild_pat = WitnessPat::wild_from_ctor(cx, ctor, ty);
        for (i, col_i) in specialized_columns.iter().enumerate() {
            // Compute witnesses for each column.
            let wits_for_col_i = collect_nonexhaustive_missing_variants(cx, col_i)?;
            // For each witness, we build a new pattern in the shape of `ctor(_, _, wit, _, _)`,
            // adding enough wildcards to match `arity`.
            for wit in wits_for_col_i {
                let mut pat = wild_pat.clone();
                pat.fields[i] = wit;
                witnesses.push(pat);
            }
        }
    }
    Ok(witnesses)
}

pub(crate) fn lint_nonexhaustive_missing_variants<'p, 'tcx>(
    rcx: &RustcPatCtxt<'p, 'tcx>,
    arms: &[MatchArm<'p, RustcPatCtxt<'p, 'tcx>>],
    pat_column: &PatternColumn<'p, RustcPatCtxt<'p, 'tcx>>,
    scrut_ty: RevealedTy<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    if !matches!(
        rcx.tcx.lint_level_at_node(NON_EXHAUSTIVE_OMITTED_PATTERNS, rcx.match_lint_level).level,
        rustc_session::lint::Level::Allow
    ) {
        let witnesses = collect_nonexhaustive_missing_variants(rcx, pat_column)?;
        if !witnesses.is_empty() {
            // Report that a match of a `non_exhaustive` enum marked with `non_exhaustive_omitted_patterns`
            // is not exhaustive enough.
            //
            // NB: The partner lint for structs lives in `compiler/rustc_hir_analysis/src/check/pat.rs`.
            rcx.tcx.emit_node_span_lint(
                NON_EXHAUSTIVE_OMITTED_PATTERNS,
                rcx.match_lint_level,
                rcx.scrut_span,
                NonExhaustiveOmittedPattern {
                    scrut_ty: scrut_ty.inner(),
                    uncovered: Uncovered::new(rcx.scrut_span, rcx, witnesses),
                },
            );
        }
    } else {
        // We used to allow putting the `#[allow(non_exhaustive_omitted_patterns)]` on a match
        // arm. This no longer makes sense so we warn users, to avoid silently breaking their
        // usage of the lint.
        for arm in arms {
            let LevelAndSource { level, src, .. } =
                rcx.tcx.lint_level_at_node(NON_EXHAUSTIVE_OMITTED_PATTERNS, arm.arm_data);
            if !matches!(level, rustc_session::lint::Level::Allow) {
                let decorator = NonExhaustiveOmittedPatternLintOnArm {
                    lint_span: src.span(),
                    suggest_lint_on_match: rcx.whole_match_span.map(|span| span.shrink_to_lo()),
                    lint_level: level.as_str(),
                    lint_name: "non_exhaustive_omitted_patterns",
                };

                use rustc_errors::LintDiagnostic;
                let mut err = rcx.tcx.dcx().struct_span_warn(arm.pat.data().span, "");
                decorator.decorate_lint(&mut err);
                err.emit();
            }
        }
    }
    Ok(())
}

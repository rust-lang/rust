//! Contains checks that must be run to validate matches before performing usefulness analysis.

use crate::constructor::Constructor::*;
use crate::pat_column::PatternColumn;
use crate::{MatchArm, PatCx};

/// Validate that deref patterns and normal constructors aren't used to match on the same place.
pub(crate) fn detect_mixed_deref_pat_ctors<'p, Cx: PatCx>(
    cx: &Cx,
    arms: &[MatchArm<'p, Cx>],
) -> Result<(), Cx::Error> {
    let pat_column = PatternColumn::new(arms);
    detect_mixed_deref_pat_ctors_inner(cx, &pat_column)
}

fn detect_mixed_deref_pat_ctors_inner<'p, Cx: PatCx>(
    cx: &Cx,
    column: &PatternColumn<'p, Cx>,
) -> Result<(), Cx::Error> {
    let Some(ty) = column.head_ty() else {
        return Ok(());
    };

    // Check for a mix of deref patterns and normal constructors.
    let mut deref_pat = None;
    let mut normal_pat = None;
    for pat in column.iter() {
        match pat.ctor() {
            // The analysis can handle mixing deref patterns with wildcards and opaque patterns.
            Wildcard | Opaque(_) => {}
            DerefPattern(_) => deref_pat = Some(pat),
            // Nothing else can be compared to deref patterns in `Constructor::is_covered_by`.
            _ => normal_pat = Some(pat),
        }
    }
    if let Some(deref_pat) = deref_pat
        && let Some(normal_pat) = normal_pat
    {
        return Err(cx.report_mixed_deref_pat_ctors(deref_pat, normal_pat));
    }

    // Specialize and recurse into the patterns' fields.
    let set = column.analyze_ctors(cx, &ty)?;
    for ctor in set.present {
        for specialized_column in column.specialize(cx, &ty, &ctor).iter() {
            detect_mixed_deref_pat_ctors_inner(cx, specialized_column)?;
        }
    }
    Ok(())
}

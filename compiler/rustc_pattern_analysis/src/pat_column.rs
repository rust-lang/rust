use crate::constructor::{Constructor, SplitConstructorSet};
use crate::pat::{DeconstructedPat, PatOrWild};
use crate::{MatchArm, PatCx};

/// A column of patterns in a match, where a column is the intuitive notion of "subpatterns that
/// inspect the same subvalue/place".
/// This is used to traverse patterns column-by-column for lints. Despite similarities with the
/// algorithm in [`crate::usefulness`], this does a different traversal. Notably this is linear in
/// the depth of patterns, whereas `compute_exhaustiveness_and_usefulness` is worst-case exponential
/// (exhaustiveness is NP-complete). The core difference is that we treat sub-columns separately.
///
/// This is not used in the usefulness algorithm; only in lints.
#[derive(Debug)]
pub struct PatternColumn<'p, Cx: PatCx> {
    /// This must not contain an or-pattern. `expand_and_push` takes care to expand them.
    patterns: Vec<&'p DeconstructedPat<Cx>>,
}

impl<'p, Cx: PatCx> PatternColumn<'p, Cx> {
    pub fn new(arms: &[MatchArm<'p, Cx>]) -> Self {
        let patterns = Vec::with_capacity(arms.len());
        let mut column = PatternColumn { patterns };
        for arm in arms {
            column.expand_and_push(PatOrWild::Pat(arm.pat));
        }
        column
    }
    /// Pushes a pattern onto the column, expanding any or-patterns into its subpatterns.
    /// Internal method, prefer [`PatternColumn::new`].
    fn expand_and_push(&mut self, pat: PatOrWild<'p, Cx>) {
        // We flatten or-patterns and skip algorithm-generated wildcards.
        if pat.is_or_pat() {
            self.patterns.extend(
                pat.flatten_or_pat().into_iter().filter_map(|pat_or_wild| pat_or_wild.as_pat()),
            )
        } else if let Some(pat) = pat.as_pat() {
            self.patterns.push(pat)
        }
    }

    pub fn head_ty(&self) -> Option<&Cx::Ty> {
        self.patterns.first().map(|pat| pat.ty())
    }
    pub fn iter(&self) -> impl Iterator<Item = &'p DeconstructedPat<Cx>> {
        self.patterns.iter().copied()
    }

    /// Do constructor splitting on the constructors of the column.
    pub fn analyze_ctors(
        &self,
        cx: &Cx,
        ty: &Cx::Ty,
    ) -> Result<SplitConstructorSet<Cx>, Cx::Error> {
        let column_ctors = self.patterns.iter().map(|p| p.ctor());
        let ctors_for_ty = cx.ctors_for_ty(ty)?;
        Ok(ctors_for_ty.split(column_ctors))
    }

    /// Does specialization: given a constructor, this takes the patterns from the column that match
    /// the constructor, and outputs their fields.
    /// This returns one column per field of the constructor. They usually all have the same length
    /// (the number of patterns in `self` that matched `ctor`), except that we expand or-patterns
    /// which may change the lengths.
    pub fn specialize(
        &self,
        cx: &Cx,
        ty: &Cx::Ty,
        ctor: &Constructor<Cx>,
    ) -> Vec<PatternColumn<'p, Cx>> {
        let arity = ctor.arity(cx, ty);
        if arity == 0 {
            return Vec::new();
        }

        // We specialize the column by `ctor`. This gives us `arity`-many columns of patterns. These
        // columns may have different lengths in the presence of or-patterns (this is why we can't
        // reuse `Matrix`).
        let mut specialized_columns: Vec<_> =
            (0..arity).map(|_| Self { patterns: Vec::new() }).collect();
        let relevant_patterns =
            self.patterns.iter().filter(|pat| ctor.is_covered_by(cx, pat.ctor()).unwrap_or(false));
        for pat in relevant_patterns {
            let specialized = pat.specialize(ctor, arity);
            for (subpat, column) in specialized.into_iter().zip(&mut specialized_columns) {
                column.expand_and_push(subpat);
            }
        }
        specialized_columns
    }
}

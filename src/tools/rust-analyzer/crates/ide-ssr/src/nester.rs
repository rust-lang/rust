//! Converts a flat collection of matches into a nested form suitable for replacement. When there
//! are multiple matches for a node, or that overlap, priority is given to the earlier rule. Nested
//! matches are only permitted if the inner match is contained entirely within a placeholder of an
//! outer match.
//!
//! For example, if our search pattern is `foo(foo($a))` and the code had `foo(foo(foo(foo(42))))`,
//! then we'll get 3 matches, however only the outermost and innermost matches can be accepted. The
//! middle match would take the second `foo` from the outer match.

use ide_db::FxHashMap;
use syntax::SyntaxNode;

use crate::{Match, SsrMatches};

pub(crate) fn nest_and_remove_collisions(
    mut matches: Vec<Match>,
    sema: &hir::Semantics<'_, ide_db::RootDatabase>,
) -> SsrMatches {
    // We sort the matches by depth then by rule index. Sorting by depth means that by the time we
    // see a match, any parent matches or conflicting matches will have already been seen. Sorting
    // by rule_index means that if there are two matches for the same node, the rule added first
    // will take precedence.
    matches.sort_by(|a, b| a.depth.cmp(&b.depth).then_with(|| a.rule_index.cmp(&b.rule_index)));
    let mut collector = MatchCollector::default();
    for m in matches {
        collector.add_match(m, sema);
    }
    collector.into()
}

#[derive(Default)]
struct MatchCollector {
    matches_by_node: FxHashMap<SyntaxNode, Match>,
}

impl MatchCollector {
    /// Attempts to add `m` to matches. If it conflicts with an existing match, it is discarded. If
    /// it is entirely within the a placeholder of an existing match, then it is added as a child
    /// match of the existing match.
    fn add_match(&mut self, m: Match, sema: &hir::Semantics<'_, ide_db::RootDatabase>) {
        let matched_node = m.matched_node.clone();
        if let Some(existing) = self.matches_by_node.get_mut(&matched_node) {
            try_add_sub_match(m, existing, sema);
            return;
        }
        for ancestor in sema.ancestors_with_macros(m.matched_node.clone()) {
            if let Some(existing) = self.matches_by_node.get_mut(&ancestor) {
                try_add_sub_match(m, existing, sema);
                return;
            }
        }
        self.matches_by_node.insert(matched_node, m);
    }
}

/// Attempts to add `m` as a sub-match of `existing`.
fn try_add_sub_match(
    m: Match,
    existing: &mut Match,
    sema: &hir::Semantics<'_, ide_db::RootDatabase>,
) {
    for p in existing.placeholder_values.values_mut() {
        // Note, no need to check if p.range.file is equal to m.range.file, since we
        // already know we're within `existing`.
        if p.range.range.contains_range(m.range.range) {
            // Convert the inner matches in `p` into a temporary MatchCollector. When
            // we're done, we then convert it back into an SsrMatches. If we expected
            // lots of inner matches, it might be worthwhile keeping a MatchCollector
            // around for each placeholder match. However we expect most placeholder
            // will have 0 and a few will have 1. More than that should hopefully be
            // exceptional.
            let mut collector = MatchCollector::default();
            for m in std::mem::take(&mut p.inner_matches.matches) {
                collector.matches_by_node.insert(m.matched_node.clone(), m);
            }
            collector.add_match(m, sema);
            p.inner_matches = collector.into();
            break;
        }
    }
}

impl From<MatchCollector> for SsrMatches {
    fn from(mut match_collector: MatchCollector) -> Self {
        let mut matches = SsrMatches::default();
        for (_, m) in match_collector.matches_by_node.drain() {
            matches.matches.push(m);
        }
        matches.matches.sort_by(|a, b| {
            // Order matches by file_id then by start range. This should be sufficient since ranges
            // shouldn't be overlapping.
            a.range
                .file_id
                .cmp(&b.range.file_id)
                .then_with(|| a.range.range.start().cmp(&b.range.range.start()))
        });
        matches
    }
}

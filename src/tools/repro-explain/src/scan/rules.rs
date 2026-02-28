use crate::model::{RuleHit, SourceLocus};

pub fn top_hits(hits: &[RuleHit], limit: usize) -> Vec<RuleHit> {
    let mut ranked = hits.to_vec();
    ranked.sort_by(|a, b| {
        strength_rank(&b.strength)
            .cmp(&strength_rank(&a.strength))
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
    });
    ranked.truncate(limit);
    ranked
}

pub fn primary_locus(hits: &[RuleHit]) -> Option<SourceLocus> {
    top_hits(hits, 1).into_iter().next().map(|hit| SourceLocus { path: hit.path, line: hit.line })
}

pub fn fix_hint(hits: &[RuleHit]) -> Option<String> {
    top_hits(hits, 1).into_iter().next().map(|hit| hit.fix_hint)
}

fn strength_rank(s: &str) -> i32 {
    match s {
        "high" => 3,
        "medium" => 2,
        "low" => 1,
        _ => 0,
    }
}

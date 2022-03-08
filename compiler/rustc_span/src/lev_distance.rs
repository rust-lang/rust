//! Levenshtein distances.
//!
//! The [Levenshtein distance] is a metric for measuring the difference between two strings.
//!
//! [Levenshtein distance]: https://en.wikipedia.org/wiki/Levenshtein_distance

use crate::symbol::Symbol;
use std::cmp;

#[cfg(test)]
mod tests;

/// Finds the Levenshtein distance between two strings.
///
/// Returns None if the distance exceeds the limit.
pub fn lev_distance(a: &str, b: &str, limit: usize) -> Option<usize> {
    let n = a.chars().count();
    let m = b.chars().count();
    let min_dist = if n < m { m - n } else { n - m };

    if min_dist > limit {
        return None;
    }
    if n == 0 || m == 0 {
        return (min_dist <= limit).then_some(min_dist);
    }

    let mut dcol: Vec<_> = (0..=m).collect();

    for (i, sc) in a.chars().enumerate() {
        let mut current = i;
        dcol[0] = current + 1;

        for (j, tc) in b.chars().enumerate() {
            let next = dcol[j + 1];
            if sc == tc {
                dcol[j + 1] = current;
            } else {
                dcol[j + 1] = cmp::min(current, next);
                dcol[j + 1] = cmp::min(dcol[j + 1], dcol[j]) + 1;
            }
            current = next;
        }
    }

    (dcol[m] <= limit).then_some(dcol[m])
}

/// Finds the best match for a given word in the given iterator.
///
/// As a loose rule to avoid the obviously incorrect suggestions, it takes
/// an optional limit for the maximum allowable edit distance, which defaults
/// to one-third of the given word.
///
/// Besides Levenshtein, we use case insensitive comparison to improve accuracy
/// on an edge case with a lower(upper)case letters mismatch.
#[cold]
pub fn find_best_match_for_name(
    candidates: &[Symbol],
    lookup: Symbol,
    dist: Option<usize>,
) -> Option<Symbol> {
    let lookup = lookup.as_str();
    let lookup_uppercase = lookup.to_uppercase();

    // Priority of matches:
    // 1. Exact case insensitive match
    // 2. Levenshtein distance match
    // 3. Sorted word match
    if let Some(c) = candidates.iter().find(|c| c.as_str().to_uppercase() == lookup_uppercase) {
        return Some(*c);
    }

    let mut dist = dist.unwrap_or_else(|| cmp::max(lookup.len(), 3) / 3);
    let mut best = None;
    for c in candidates {
        match lev_distance(lookup, c.as_str(), dist) {
            Some(0) => return Some(*c),
            Some(d) => {
                dist = d - 1;
                best = Some(*c);
            }
            None => {}
        }
    }
    if best.is_some() {
        return best;
    }

    find_match_by_sorted_words(candidates, lookup)
}

fn find_match_by_sorted_words(iter_names: &[Symbol], lookup: &str) -> Option<Symbol> {
    iter_names.iter().fold(None, |result, candidate| {
        if sort_by_words(candidate.as_str()) == sort_by_words(lookup) {
            Some(*candidate)
        } else {
            result
        }
    })
}

fn sort_by_words(name: &str) -> String {
    let mut split_words: Vec<&str> = name.split('_').collect();
    // We are sorting primitive &strs and can use unstable sort here.
    split_words.sort_unstable();
    split_words.join("_")
}

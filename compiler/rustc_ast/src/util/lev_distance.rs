// FIXME(Centril): Move to rustc_span?

use rustc_span::symbol::Symbol;
use std::cmp;

#[cfg(test)]
mod tests;

/// Finds the Levenshtein distance between two strings
pub fn lev_distance(a: &str, b: &str) -> usize {
    // cases which don't require further computation
    if a.is_empty() {
        return b.chars().count();
    } else if b.is_empty() {
        return a.chars().count();
    }

    let mut dcol: Vec<_> = (0..=b.len()).collect();
    let mut t_last = 0;

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
            t_last = j;
        }
    }
    dcol[t_last + 1]
}

/// Finds the best match for a given word in the given iterator
///
/// As a loose rule to avoid the obviously incorrect suggestions, it takes
/// an optional limit for the maximum allowable edit distance, which defaults
/// to one-third of the given word.
///
/// Besides Levenshtein, we use case insensitive comparison to improve accuracy on an edge case with
/// a lower(upper)case letters mismatch.
pub fn find_best_match_for_name<'a, T>(
    iter_names: T,
    lookup: Symbol,
    dist: Option<usize>,
) -> Option<Symbol>
where
    T: Iterator<Item = &'a Symbol>,
{
    let lookup = &lookup.as_str();
    let max_dist = dist.unwrap_or_else(|| cmp::max(lookup.len(), 3) / 3);
    let name_vec: Vec<&Symbol> = iter_names.collect();

    let (case_insensitive_match, levenshtein_match) = name_vec
        .iter()
        .filter_map(|&name| {
            let dist = lev_distance(lookup, &name.as_str());
            if dist <= max_dist { Some((name, dist)) } else { None }
        })
        // Here we are collecting the next structure:
        // (case_insensitive_match, (levenshtein_match, levenshtein_distance))
        .fold((None, None), |result, (candidate, dist)| {
            (
                if candidate.as_str().to_uppercase() == lookup.to_uppercase() {
                    Some(candidate)
                } else {
                    result.0
                },
                match result.1 {
                    None => Some((candidate, dist)),
                    Some((c, d)) => Some(if dist < d { (candidate, dist) } else { (c, d) }),
                },
            )
        });
    // Priority of matches:
    // 1. Exact case insensitive match
    // 2. Levenshtein distance match
    // 3. Sorted word match
    if let Some(candidate) = case_insensitive_match {
        Some(*candidate)
    } else if levenshtein_match.is_some() {
        levenshtein_match.map(|(candidate, _)| *candidate)
    } else {
        find_match_by_sorted_words(name_vec, lookup)
    }
}

fn find_match_by_sorted_words<'a>(iter_names: Vec<&'a Symbol>, lookup: &str) -> Option<Symbol> {
    iter_names.iter().fold(None, |result, candidate| {
        if sort_by_words(&candidate.as_str()) == sort_by_words(lookup) {
            Some(**candidate)
        } else {
            result
        }
    })
}

fn sort_by_words(name: &str) -> String {
    let mut split_words: Vec<&str> = name.split('_').collect();
    // We are sorting primitive &strs and can use unstable sort here
    split_words.sort_unstable();
    split_words.join("_")
}

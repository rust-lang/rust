//! Edit distances.
//!
//! The [edit distance] is a metric for measuring the difference between two strings.
//!
//! [edit distance]: https://en.wikipedia.org/wiki/Edit_distance

// The current implementation is the restricted Damerau-Levenshtein algorithm. It is restricted
// because it does not permit modifying characters that have already been transposed. The specific
// algorithm should not matter to the caller of the methods, which is why it is not noted in the
// documentation.

use std::{cmp, mem};

use crate::Symbol;

#[cfg(test)]
mod tests;

/// Finds the [edit distance] between two strings.
///
/// Returns `None` if the distance exceeds the limit.
///
/// [edit distance]: https://en.wikipedia.org/wiki/Edit_distance
pub fn edit_distance(a: &str, b: &str, limit: usize) -> Option<usize> {
    let mut a = &a.chars().collect::<Vec<_>>()[..];
    let mut b = &b.chars().collect::<Vec<_>>()[..];

    // Ensure that `b` is the shorter string, minimizing memory use.
    if a.len() < b.len() {
        mem::swap(&mut a, &mut b);
    }

    let min_dist = a.len() - b.len();
    // If we know the limit will be exceeded, we can return early.
    if min_dist > limit {
        return None;
    }

    // Strip common prefix.
    while let Some(((b_char, b_rest), (a_char, a_rest))) = b.split_first().zip(a.split_first())
        && a_char == b_char
    {
        a = a_rest;
        b = b_rest;
    }
    // Strip common suffix.
    while let Some(((b_char, b_rest), (a_char, a_rest))) = b.split_last().zip(a.split_last())
        && a_char == b_char
    {
        a = a_rest;
        b = b_rest;
    }

    // If either string is empty, the distance is the length of the other.
    // We know that `b` is the shorter string, so we don't need to check `a`.
    if b.len() == 0 {
        return Some(min_dist);
    }

    let mut prev_prev = vec![usize::MAX; b.len() + 1];
    let mut prev = (0..=b.len()).collect::<Vec<_>>();
    let mut current = vec![0; b.len() + 1];

    // row by row
    for i in 1..=a.len() {
        current[0] = i;
        let a_idx = i - 1;

        // column by column
        for j in 1..=b.len() {
            let b_idx = j - 1;

            // There is no cost to substitute a character with itself.
            let substitution_cost = if a[a_idx] == b[b_idx] { 0 } else { 1 };

            current[j] = cmp::min(
                // deletion
                prev[j] + 1,
                cmp::min(
                    // insertion
                    current[j - 1] + 1,
                    // substitution
                    prev[j - 1] + substitution_cost,
                ),
            );

            if (i > 1) && (j > 1) && (a[a_idx] == b[b_idx - 1]) && (a[a_idx - 1] == b[b_idx]) {
                // transposition
                current[j] = cmp::min(current[j], prev_prev[j - 2] + 1);
            }
        }

        // Rotate the buffers, reusing the memory.
        [prev_prev, prev, current] = [prev, current, prev_prev];
    }

    // `prev` because we already rotated the buffers.
    let distance = prev[b.len()];
    (distance <= limit).then_some(distance)
}

/// Provides a word similarity score between two words that accounts for substrings being more
/// meaningful than a typical edit distance. The lower the score, the closer the match. 0 is an
/// identical match.
///
/// Uses the edit distance between the two strings and removes the cost of the length difference.
/// If this is 0 then it is either a substring match or a full word match, in the substring match
/// case we detect this and return `1`. To prevent finding meaningless substrings, eg. "in" in
/// "shrink", we only perform this subtraction of length difference if one of the words is not
/// greater than twice the length of the other. For cases where the words are close in size but not
/// an exact substring then the cost of the length difference is discounted by half.
///
/// Returns `None` if the distance exceeds the limit.
pub fn edit_distance_with_substrings(a: &str, b: &str, limit: usize) -> Option<usize> {
    let n = a.chars().count();
    let m = b.chars().count();

    // Check one isn't less than half the length of the other. If this is true then there is a
    // big difference in length.
    let big_len_diff = (n * 2) < m || (m * 2) < n;
    let len_diff = m.abs_diff(n);
    let distance = edit_distance(a, b, limit + len_diff)?;

    // This is the crux, subtracting length difference means exact substring matches will now be 0
    let score = distance - len_diff;

    // If the score is 0 but the words have different lengths then it's a substring match not a full
    // word match
    let score = if score == 0 && len_diff > 0 && !big_len_diff {
        1 // Exact substring match, but not a total word match so return non-zero
    } else if !big_len_diff {
        // Not a big difference in length, discount cost of length difference
        score + len_diff.div_ceil(2)
    } else {
        // A big difference in length, add back the difference in length to the score
        score + len_diff
    };

    (score <= limit).then_some(score)
}

/// Finds the best match for given word in the given iterator where substrings are meaningful.
///
/// A version of [`find_best_match_for_name`] that uses [`edit_distance_with_substrings`] as the
/// score for word similarity. This takes an optional distance limit which defaults to one-third of
/// the given word.
///
/// We use case insensitive comparison to improve accuracy on an edge case with a lower(upper)case
/// letters mismatch.
pub fn find_best_match_for_name_with_substrings(
    candidates: &[Symbol],
    lookup: Symbol,
    dist: Option<usize>,
) -> Option<Symbol> {
    find_best_match_for_name_impl(true, candidates, lookup, dist)
}

/// Finds the best match for a given word in the given iterator.
///
/// As a loose rule to avoid the obviously incorrect suggestions, it takes
/// an optional limit for the maximum allowable edit distance, which defaults
/// to one-third of the given word.
///
/// We use case insensitive comparison to improve accuracy on an edge case with a lower(upper)case
/// letters mismatch.
pub fn find_best_match_for_name(
    candidates: &[Symbol],
    lookup: Symbol,
    dist: Option<usize>,
) -> Option<Symbol> {
    find_best_match_for_name_impl(false, candidates, lookup, dist)
}

/// Find the best match for multiple words
///
/// This function is intended for use when the desired match would never be
/// returned due to a substring in `lookup` which is superfluous.
///
/// For example, when looking for the closest lint name to `clippy:missing_docs`,
/// we would find `clippy::erasing_op`, despite `missing_docs` existing and being a better suggestion.
/// `missing_docs` would have a larger edit distance because it does not contain the `clippy` tool prefix.
/// In order to find `missing_docs`, this function takes multiple lookup strings, computes the best match
/// for each and returns the match which had the lowest edit distance. In our example, `clippy:missing_docs` and
/// `missing_docs` would be `lookups`, enabling `missing_docs` to be the best match, as desired.
pub fn find_best_match_for_names(
    candidates: &[Symbol],
    lookups: &[Symbol],
    dist: Option<usize>,
) -> Option<Symbol> {
    lookups
        .iter()
        .map(|s| (s, find_best_match_for_name_impl(false, candidates, *s, dist)))
        .filter_map(|(s, r)| r.map(|r| (s, r)))
        .min_by(|(s1, r1), (s2, r2)| {
            let d1 = edit_distance(s1.as_str(), r1.as_str(), usize::MAX).unwrap();
            let d2 = edit_distance(s2.as_str(), r2.as_str(), usize::MAX).unwrap();
            d1.cmp(&d2)
        })
        .map(|(_, r)| r)
}

#[cold]
fn find_best_match_for_name_impl(
    use_substring_score: bool,
    candidates: &[Symbol],
    lookup_symbol: Symbol,
    dist: Option<usize>,
) -> Option<Symbol> {
    let lookup = lookup_symbol.as_str();
    let lookup_uppercase = lookup.to_uppercase();

    // Priority of matches:
    // 1. Exact case insensitive match
    // 2. Edit distance match
    // 3. Sorted word match
    if let Some(c) = candidates.iter().find(|c| c.as_str().to_uppercase() == lookup_uppercase) {
        return Some(*c);
    }

    // `fn edit_distance()` use `chars()` to calculate edit distance, so we must
    // also use `chars()` (and not `str::len()`) to calculate length here.
    let lookup_len = lookup.chars().count();

    let mut dist = dist.unwrap_or_else(|| cmp::max(lookup_len, 3) / 3);
    let mut best = None;
    // store the candidates with the same distance, only for `use_substring_score` current.
    let mut next_candidates = vec![];
    for c in candidates {
        match if use_substring_score {
            edit_distance_with_substrings(lookup, c.as_str(), dist)
        } else {
            edit_distance(lookup, c.as_str(), dist)
        } {
            Some(0) => return Some(*c),
            Some(d) => {
                if use_substring_score {
                    if d < dist {
                        dist = d;
                        next_candidates.clear();
                    } else {
                        // `d == dist` here, we need to store the candidates with the same distance
                        // so we won't decrease the distance in the next loop.
                    }
                    next_candidates.push(*c);
                } else {
                    dist = d - 1;
                }
                best = Some(*c);
            }
            None => {}
        }
    }

    // We have a tie among several candidates, try to select the best among them ignoring substrings.
    // For example, the candidates list `force_capture`, `capture`, and user inputted `forced_capture`,
    // we select `force_capture` with a extra round of edit distance calculation.
    if next_candidates.len() > 1 {
        debug_assert!(use_substring_score);
        best = find_best_match_for_name_impl(
            false,
            &next_candidates,
            lookup_symbol,
            Some(lookup.len()),
        );
    }
    if best.is_some() {
        return best;
    }

    find_match_by_sorted_words(candidates, lookup)
}

fn find_match_by_sorted_words(iter_names: &[Symbol], lookup: &str) -> Option<Symbol> {
    let lookup_sorted_by_words = sort_by_words(lookup);
    iter_names.iter().fold(None, |result, candidate| {
        if sort_by_words(candidate.as_str()) == lookup_sorted_by_words {
            Some(*candidate)
        } else {
            result
        }
    })
}

fn sort_by_words(name: &str) -> Vec<&str> {
    let mut split_words: Vec<&str> = name.split('_').collect();
    // We are sorting primitive &strs and can use unstable sort here.
    split_words.sort_unstable();
    split_words
}

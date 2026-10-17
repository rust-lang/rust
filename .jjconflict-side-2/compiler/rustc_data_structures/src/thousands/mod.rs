//! This is a bare-bones alternative to the `thousands` crate on crates.io, for
//! printing large numbers in a readable fashion.

#[cfg(test)]
mod tests;

fn format_with_underscores(mut s: String) -> String {
    // Ignore a leading '-'.
    let start = if s.starts_with('-') { 1 } else { 0 };

    // Stop after the first non-digit, e.g. '.' or 'e' for floats.
    let non_digit = s[start..].find(|c: char| !c.is_digit(10));
    let end = if let Some(non_digit) = non_digit { start + non_digit } else { s.len() };

    // Insert underscores within `start..end`.
    let mut i = end;
    while i > start + 3 {
        i -= 3;
        s.insert(i, '_');
    }
    s
}

/// Print a `usize` with underscore separators.
pub fn usize_with_underscores(n: usize) -> String {
    format_with_underscores(format!("{n}"))
}

/// Print an `isize` with underscore separators.
pub fn isize_with_underscores(n: isize) -> String {
    format_with_underscores(format!("{n}"))
}

/// Print an `f64` with precision 1 (one decimal place) and underscore separators.
pub fn f64p1_with_underscores(n: f64) -> String {
    format_with_underscores(format!("{n:.1}"))
}

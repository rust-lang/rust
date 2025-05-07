//! This is an extremely bare-bones alternative to the `thousands` crate on
//! crates.io, for printing large numbers in a readable fashion.

#[cfg(test)]
mod tests;

// Converts the number to a string, with underscores as the thousands separator.
pub fn format_with_underscores(n: usize) -> String {
    let mut s = n.to_string();
    let mut i = s.len();
    while i > 3 {
        i -= 3;
        s.insert(i, '_');
    }
    s
}

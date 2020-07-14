#![warn(clippy::unnested_or_patterns)]

// Test that `unnested_or_patterns` does not trigger without enabling `or_patterns`
fn main() {
    if let (0, 1) | (0, 2) | (0, 3) = (0, 0) {}
}

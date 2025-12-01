//! Regression test for <https://github.com/rust-lang/rust/issues/50688>

fn main() {
    [1; || {}]; //~ ERROR mismatched types
}

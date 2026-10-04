//! Very basic smoke test to make sure `run_make_support::env::jobs` at least does not panic.

fn main() {
    println!("{}", run_make_support::env::jobs());
}

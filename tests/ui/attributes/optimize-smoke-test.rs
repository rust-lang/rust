//! Basic smoke test for `#[optimize(..)]` attributes.
//@ run-pass

#[optimize(speed)]
fn optimized_speed() -> i32 {
    42
}

#[optimize(size)]
fn optimized_size() -> i32 {
    42
}

#[optimize(none)]
fn optimized_none() -> i32 {
    42
}

fn main() {
    assert_eq!(optimized_speed(), 42);
    assert_eq!(optimized_size(), 42);
    assert_eq!(optimized_none(), 42);
}

// run-rustfix

#![feature(const_result)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(clippy::match_like_matches_macro, unused)]

// Test that results are linted with the feature enabled.

const fn issue_5697() {
    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {}

    while let Ok(_) = Ok::<i32, i32>(10) {}

    while let Err(_) = Ok::<i32, i32>(10) {}

    match Ok::<i32, i32>(42) {
        Ok(_) => true,
        Err(_) => false,
    };

    match Err::<i32, i32>(42) {
        Ok(_) => false,
        Err(_) => true,
    };

    // These should not be linted until `const_option` is implemented.
    // See https://github.com/rust-lang/rust/issues/67441

    if let Some(_) = Some(42) {}

    if let None = None::<()> {}

    while let Some(_) = Some(42) {}

    while let None = None::<()> {}

    match Some(42) {
        Some(_) => true,
        None => false,
    };

    match None::<()> {
        Some(_) => false,
        None => true,
    };
}

fn main() {}

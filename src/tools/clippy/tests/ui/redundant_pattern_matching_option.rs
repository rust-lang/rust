// run-rustfix

#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(
    clippy::unit_arg,
    unused_must_use,
    clippy::needless_bool,
    clippy::match_like_matches_macro,
    deprecated
)]

fn main() {
    if let None = None::<()> {}

    if let Some(_) = Some(42) {}

    if let Some(_) = Some(42) {
        foo();
    } else {
        bar();
    }

    while let Some(_) = Some(42) {}

    while let None = Some(42) {}

    while let None = None::<()> {}

    let mut v = vec![1, 2, 3];
    while let Some(_) = v.pop() {
        foo();
    }

    if None::<i32>.is_none() {}

    if Some(42).is_some() {}

    match Some(42) {
        Some(_) => true,
        None => false,
    };

    match None::<()> {
        Some(_) => false,
        None => true,
    };

    let _ = match None::<()> {
        Some(_) => false,
        None => true,
    };

    let opt = Some(false);
    let x = if let Some(_) = opt { true } else { false };
    takes_bool(x);

    issue6067();

    let _ = if let Some(_) = gen_opt() {
        1
    } else if let None = gen_opt() {
        2
    } else {
        3
    };
}

fn gen_opt() -> Option<()> {
    None
}

fn takes_bool(_: bool) {}

fn foo() {}

fn bar() {}

// Methods that are unstable const should not be suggested within a const context, see issue #5697.
// However, in Rust 1.48.0 the methods `is_some` and `is_none` of `Option` were stabilized as const,
// so the following should be linted.
const fn issue6067() {
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

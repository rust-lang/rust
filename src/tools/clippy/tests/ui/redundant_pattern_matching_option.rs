//@run-rustfix

#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(
    unused_must_use,
    clippy::needless_bool,
    clippy::match_like_matches_macro,
    clippy::equatable_if_let,
    clippy::if_same_then_else
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
    let _ = if let Some(_) = opt { true } else { false };

    issue6067();
    issue10726();
    issue10803();

    let _ = if let Some(_) = gen_opt() {
        1
    } else if let None = gen_opt() {
        2
    } else {
        3
    };

    if let Some(..) = gen_opt() {}
}

fn gen_opt() -> Option<()> {
    None
}

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

#[allow(clippy::deref_addrof, dead_code, clippy::needless_borrow)]
fn issue7921() {
    if let None = *(&None::<()>) {}
    if let None = *&None::<()> {}
}

fn issue10726() {
    let x = Some(42);

    match x {
        Some(_) => true,
        _ => false,
    };

    match x {
        None => true,
        _ => false,
    };

    match x {
        Some(_) => false,
        _ => true,
    };

    match x {
        None => false,
        _ => true,
    };

    // Don't lint
    match x {
        Some(21) => true,
        _ => false,
    };
}

fn issue10803() {
    let x = Some(42);

    let _ = matches!(x, Some(_));

    let _ = matches!(x, None);

    // Don't lint
    let _ = matches!(x, Some(16));
}

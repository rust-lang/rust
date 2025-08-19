#![feature(if_let_guard)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(
    clippy::needless_bool,
    clippy::needless_if,
    clippy::match_like_matches_macro,
    clippy::equatable_if_let,
    clippy::if_same_then_else
)]

fn issue_11174<T>(boolean: bool, maybe_some: Option<T>) -> bool {
    matches!(maybe_some, None if !boolean)
    //~^ redundant_pattern_matching
}

fn issue_11174_edge_cases<T>(boolean: bool, boolean2: bool, maybe_some: Option<T>) {
    let _ = matches!(maybe_some, None if boolean || boolean2); // guard needs parentheses
    //
    //~^^ redundant_pattern_matching
    let _ = match maybe_some {
        // can't use `matches!` here
        // because `expr` metavars in macros don't allow let exprs
        None if let Some(x) = Some(0)
            && x > 5 =>
        {
            true
        },
        _ => false,
    };
}

fn main() {
    if let None = None::<()> {}
    //~^ redundant_pattern_matching

    if let Some(_) = Some(42) {}
    //~^ redundant_pattern_matching

    if let Some(_) = Some(42) {
        //~^ redundant_pattern_matching
        foo();
    } else {
        bar();
    }

    while let Some(_) = Some(42) {}
    //~^ redundant_pattern_matching

    while let None = Some(42) {}
    //~^ redundant_pattern_matching

    while let None = None::<()> {}
    //~^ redundant_pattern_matching

    let mut v = vec![1, 2, 3];
    while let Some(_) = v.pop() {
        //~^ redundant_pattern_matching
        foo();
    }

    if None::<i32>.is_none() {}

    if Some(42).is_some() {}

    match Some(42) {
        //~^ redundant_pattern_matching
        Some(_) => true,
        None => false,
    };

    match None::<()> {
        //~^ redundant_pattern_matching
        Some(_) => false,
        None => true,
    };

    let _ = match None::<()> {
        //~^ redundant_pattern_matching
        Some(_) => false,
        None => true,
    };

    let opt = Some(false);
    let _ = if let Some(_) = opt { true } else { false };
    //~^ redundant_pattern_matching

    issue6067();
    issue10726();
    issue10803();

    let _ = if let Some(_) = gen_opt() {
        //~^ redundant_pattern_matching
        1
    } else if let None = gen_opt() {
        //~^ redundant_pattern_matching
        2
    } else {
        3
    };

    if let Some(..) = gen_opt() {}
    //~^ redundant_pattern_matching
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
    //~^ redundant_pattern_matching

    if let None = None::<()> {}
    //~^ redundant_pattern_matching

    while let Some(_) = Some(42) {}
    //~^ redundant_pattern_matching

    while let None = None::<()> {}
    //~^ redundant_pattern_matching

    match Some(42) {
        //~^ redundant_pattern_matching
        Some(_) => true,
        None => false,
    };

    match None::<()> {
        //~^ redundant_pattern_matching
        Some(_) => false,
        None => true,
    };
}

#[allow(clippy::deref_addrof, dead_code, clippy::needless_borrow)]
fn issue7921() {
    if let None = *(&None::<()>) {}
    //~^ redundant_pattern_matching
    if let None = *&None::<()> {}
    //~^ redundant_pattern_matching
}

fn issue10726() {
    let x = Some(42);

    match x {
        //~^ redundant_pattern_matching
        Some(_) => true,
        _ => false,
    };

    match x {
        //~^ redundant_pattern_matching
        None => true,
        _ => false,
    };

    match x {
        //~^ redundant_pattern_matching
        Some(_) => false,
        _ => true,
    };

    match x {
        //~^ redundant_pattern_matching
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
    //~^ redundant_pattern_matching

    let _ = matches!(x, None);
    //~^ redundant_pattern_matching

    // Don't lint
    let _ = matches!(x, Some(16));
}

fn issue13902() {
    let x = Some(0);
    let p = &raw const x;
    unsafe {
        let _ = matches!(*p, None);
        //~^ redundant_pattern_matching
    }
}

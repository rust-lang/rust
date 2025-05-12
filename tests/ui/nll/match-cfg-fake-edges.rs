// Test that we have enough false edges to avoid exposing the exact matching
// algorithm in borrow checking.

#![feature(if_let_guard)]

#[rustfmt::skip]
fn all_patterns_are_tested() {
    // Even though `x` is never actually moved out of, we don't want borrowck results to be based on
    // whether MIR lowering reveals which patterns are unreachable.
    let x = String::new();
    match true {
        _ => {},
        _ => drop(x),
    }
    // Borrowck must not know the second arm is never run.
    drop(x); //~ ERROR use of moved value

    let x = String::new();
    if let _ = true { //~ WARN irrefutable
    } else {
        drop(x)
    }
    // Borrowck must not know the else branch is never run.
    drop(x); //~ ERROR use of moved value

    let x = (String::new(), String::new());
    match x {
        (y, _) | (_, y) => (),
    }
    &x.0; //~ ERROR borrow of moved value
    // Borrowck must not know the second pattern never matches.
    &x.1; //~ ERROR borrow of moved value

    let x = (String::new(), String::new());
    let ((y, _) | (_, y)) = x;
    &x.0; //~ ERROR borrow of moved value
    // Borrowck must not know the second pattern never matches.
    &x.1; //~ ERROR borrow of moved value
}

#[rustfmt::skip]
fn guard_always_precedes_arm(y: i32) {
    // x should always be initialized, as the only way to reach the arm is
    // through the guard.
    let mut x;
    match y {
        0 | 2 if { x = 2; true } => x,
        _ => 2,
    };

    let mut x;
    match y {
        _ => 2,
        0 | 2 if { x = 2; true } => x,
    };

    let mut x;
    match y {
        0 | 2 if let Some(()) = { x = 2; Some(()) } => x,
        _ => 2,
    };
}

#[rustfmt::skip]
fn guard_may_be_skipped(y: i32) {
    // Even though x *is* always initialized, we don't want to have borrowck results be based on
    // whether MIR lowering reveals which patterns are exhaustive.
    let x;
    match y {
        _ if { x = 2; true } => {},
        // Borrowck must not know the guard is always run.
        _ => drop(x), //~ ERROR used binding `x` is possibly-uninitialized
    };

    let x;
    match y {
        _ if { x = 2; true } => 1,
        // Borrowck must not know the guard is always run.
        _ if { x; false } => 2, //~ ERROR used binding `x` isn't initialized
        _ => 3,
    };

    let x;
    match y {
        _ if let Some(()) = { x = 2; Some(()) } => 1,
        _ if let Some(()) = { x; None } => 2, //~ ERROR used binding `x` isn't initialized
        _ => 3,
    };
}

#[rustfmt::skip]
fn guard_may_be_taken(y: bool) {
    // Even though x *is* never moved before the use, we don't want to have
    // borrowck results be based on whether patterns are disjoint.
    let x = String::new();
    match y {
        false if { drop(x); true } => {},
        // Borrowck must not know the guard is not run in the `true` case.
        true => drop(x), //~ ERROR use of moved value: `x`
        false => {},
    };

    // Fine in the other order.
    let x = String::new();
    match y {
        true => drop(x),
        false if { drop(x); true } => {},
        false => {},
    };

    let x = String::new();
    match y {
        false if let Some(()) = { drop(x); Some(()) } => {},
        true => drop(x), //~ ERROR use of moved value: `x`
        false => {},
    };
}

fn main() {}

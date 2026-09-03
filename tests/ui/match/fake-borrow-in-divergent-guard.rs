//! Regression test for <https://github.com/rust-lang/rust/issues/161578>: the fake borrow on `x`
//! below was ignored previously because the fake read keeping it live was unreachable.
//@ check-pass
// TODO: this should be check-fail

#![feature(explicit_tail_calls)]

// In this first test, it's possible for the guard to fail. We need the fake borrow for soundness.

fn main() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    match x {
        Some(_) if { x = None; false } && return => {}
        // TODO: ERROR: cannot assign `x` in match guard
        Some(b) => println!("{b}"),
        None => println!("none"),
    }
}

// In the following tests, the guard can't fail, but we keep the fake borrow alive for consistency.

fn always_return_after_mutation() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    match x {
        Some(_) if { x = None; return } => {}
        // TODO: ERROR: cannot assign `x` in match guard
        Some(b) => println!("{b}"),
        None => println!("none"),
    }
}

fn always_panic_after_mutation() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    match x {
        Some(_) if { x = None; panic!() } => {}
        // TODO: ERROR: cannot assign `x` in match guard
        Some(b) => println!("{b}"),
        None => println!("none"),
    }
}

fn always_break_after_mutation() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    'b: {
        match x {
            Some(_) if { x = None; break 'b } => {}
            // TODO: ERROR: cannot assign `x` in match guard
            Some(b) => println!("{b}"),
            None => println!("none"),
        }
    }
}

fn always_continue_after_mutation() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    loop {
        match x {
            Some(_) if { x = None; continue } => {}
            // TODO: ERROR: cannot assign `x` in match guard
            Some(ref b) => println!("{b}"),
            None => println!("none"),
        }
    }
}

fn always_become_after_mutation() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    match x {
        Some(_) if { x = None; become always_become_after_mutation() } => {}
        // TODO: ERROR: cannot assign `x` in match guard
        Some(b) => println!("{b}"),
        None => println!("none"),
    }
}

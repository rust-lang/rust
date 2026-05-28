//@ revisions: pass fail
//@[pass] check-pass
//@[fail] check-fail
#![feature(never_patterns)]
#![allow(incomplete_features)]

#[derive(Copy, Clone)]
enum Void {}

fn main() {}

// The classic use for empty types.
fn safe_unwrap_result<T: Copy>(res: Result<T, Void>) {
    let Ok(_x) = res;
    let (Ok(_x) | Err(!)) = &res;
    let (Ok(_x) | Err(!)) = res.as_ref();
}

// Check we only accept `!` where we want to.
#[cfg(fail)]
fn never_pattern_typeck_fail(void: Void) {
    // Don't accept on a non-empty type.
    match () {
        !,
        //[fail]~^ ERROR: mismatched types
    }
    match (0, false) {
        !,
        //[fail]~^ ERROR: mismatched types
    }
    match (0, false) {
        (_, !),
        //[fail]~^ ERROR: mismatched types
    }
    match Some(0) {
        None => {}
        Some(!),
        //[fail]~^ ERROR: mismatched types
    }

    // Don't accept on an arbitrary type, even if there are no more branches.
    match () {
        () => {}
        !,
        //[fail]~^ ERROR: mismatched types
    }

    // Don't accept even on an empty branch.
    match None::<Void> {
        None => {}
        !,
        //[fail]~^ ERROR: mismatched types
    }
    match (&[] as &[Void]) {
        [] => {}
        !,
        //[fail]~^ ERROR: mismatched types
    }
    // Let alone if the emptiness is behind a reference.
    match None::<&Void> {
        None => {}
        !,
        //[fail]~^ ERROR: mismatched types
    }
}

#[cfg(pass)]
fn never_pattern_typeck_pass(void: Void) {
    // Participate in match ergonomics.
    match &void {
        !,
    }
    match &&void {
        !,
    }
    match &&void {
        &!,
    }
    match &None::<Void> {
        None => {}
        Some(!),
    }
    match None::<&Void> {
        None => {}
        Some(!),
    }

    // Accept on a directly empty type.
    match void {
        !,
    }
    match &void {
        &!,
    }
    match None::<Void> {
        None => {}
        Some(!),
    }
    match None::<&Void> {
        None => {}
        Some(&!),
    }
    match None::<&(u32, Void)> {
        None => {}
        Some(&(_, !)),
    }
    match (&[] as &[Void]) {
        [] => {}
        [!, ..],
    }
    // Accept on a composite empty type.
    match None::<&(u32, Void)> {
        None => {}
        Some(&!),
    }
    match None::<&(u32, Void)> {
        None => {}
        Some(!),
    }
    match None::<&Result<Void, Void>> {
        None => {}
        Some(!),
    }
}

struct Unsized {
    void: Void,
    slice: [u8],
}

#[cfg(pass)]
fn not_sized(x: &Unsized) {
    match *x {
        !,
    }
}

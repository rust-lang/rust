#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn main() {}

// The classic use for empty types.
fn safe_unwrap_result<T>(res: Result<T, Void>) {
    let Ok(_x) = res;
    // FIXME(never_patterns): These should be allowed
    let (Ok(_x) | Err(!)) = &res;
    //~^ ERROR: is not bound in all patterns
    let (Ok(_x) | Err(&!)) = res.as_ref();
    //~^ ERROR: is not bound in all patterns
}

// Check we only accept `!` where we want to.
fn never_pattern_location(void: Void) {
    // FIXME(never_patterns): Don't accept on a non-empty type.
    match Some(0) {
        None => {}
        Some(!) => {}
    }
    // FIXME(never_patterns): Don't accept on an arbitrary type, even if there are no more branches.
    match () {
        () => {}
        ! => {}
    }
    // FIXME(never_patterns): Don't accept even on an empty branch.
    match None::<Void> {
        None => {}
        ! => {}
    }
    // FIXME(never_patterns): Let alone if the emptiness is behind a reference.
    match None::<&Void> {
        None => {}
        ! => {}
    }
    // Participate in match ergonomics.
    match &void {
        ! => {}
    }
    match &&void {
        ! => {}
    }
    match &&void {
        &! => {}
    }
    match &None::<Void> {
        None => {}
        Some(!) => {}
    }
    match None::<&Void> {
        None => {}
        Some(!) => {}
    }
    // Accept on a composite empty type.
    match None::<&(u32, Void)> {
        None => {}
        Some(&!) => {}
    }
    // Accept on an simple empty type.
    match None::<Void> {
        None => {}
        Some(!) => {}
    }
    match None::<&Void> {
        None => {}
        Some(&!) => {}
    }
    match None::<&(u32, Void)> {
        None => {}
        Some(&(_, !)) => {}
    }
}

fn never_and_bindings() {
    let x: Result<bool, &(u32, Void)> = Ok(false);

    // FIXME(never_patterns): Never patterns in or-patterns don't need to share the same bindings.
    match x {
        Ok(_x) | Err(&!) => {}
        //~^ ERROR: is not bound in all patterns
    }
    let (Ok(_x) | Err(&!)) = x;
        //~^ ERROR: is not bound in all patterns

    // FIXME(never_patterns): A never pattern mustn't have bindings.
    match x {
        Ok(_) => {}
        Err(&(_b, !)) => {}
    }
    match x {
        Ok(_a) | Err(&(_b, !)) => {}
        //~^ ERROR: is not bound in all patterns
        //~| ERROR: is not bound in all patterns
    }
}

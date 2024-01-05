// check-pass
#![feature(never_patterns)]
#![feature(exhaustive_patterns)]
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
fn never_pattern_typeck(void: Void) {
    // FIXME(never_patterns): Don't accept on a non-empty type.
    match () {
        !,
    }
    match (0, false) {
        !,
    }
    match (0, false) {
        (_, !),
    }
    match Some(0) {
        None => {}
        Some(!),
    }

    // FIXME(never_patterns): Don't accept on an arbitrary type, even if there are no more branches.
    match () {
        () => {}
        !,
    }

    // FIXME(never_patterns): Don't accept even on an empty branch.
    match None::<Void> {
        None => {}
        !,
    }
    match (&[] as &[Void]) {
        [] => {}
        !,
    }
    // FIXME(never_patterns): Let alone if the emptiness is behind a reference.
    match None::<&Void> {
        None => {}
        !,
    }

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
        [!],
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

//! Test that guard patterns can see bindings already in scope and bindings introduced in their
//! subpattern, but no other bindings from the containing pattern. Also make sure bindings
//! introduced in guard patterns are visible in fn/arm/loop/etc bodies.

#![feature(guard_patterns)]
#![expect(incomplete_features)]

fn good_fn_item(((x if x) | x): bool) -> bool { x }

fn bad_fn_item_1(x: bool, ((y if x) | y): bool) {}
//~^ ERROR cannot find value `x` in this scope
fn bad_fn_item_2(((x if y) | x): bool, y: bool) {}
//~^ ERROR cannot find value `y` in this scope

fn main() {
    let ((local if local) if local) = false;

    match (true, true) {
        (x if local, y if good_fn_item(y)) => x && y,
        (x, y if x) => x && y,
        //~^ ERROR cannot find value `x` in this scope
        (x if y, y) => x && y,
        //~^ ERROR cannot find value `y` in this scope
    };

    match (true,) {
        (x @ y if x && y,) => x && y,
        (x @ (y if y),) => x && y,
        (x @ (y if x),) => x && y,
        //~^ ERROR cannot find value `x` in this scope
    };

    match (Ok(true),) {
        ((Ok(x) | Err(x)) if good_fn_item(x),) => x,
        ((Ok(x) if local) | (Err(x) if good_fn_item(x)),) => x,
        ((Ok(x if x) if x) | (Err(x if x) if x) if x,) if x => x,
        ((Ok(x) if y) | (Err(y) if x),) => x && y,
        //~^ ERROR variable `x` is not bound in all patterns
        //~| ERROR variable `y` is not bound in all patterns
        //~| ERROR cannot find value `x` in this scope
        //~| ERROR cannot find value `y` in this scope
    };

    let (_ if nonexistent) = true;
    //~^ ERROR cannot find value `nonexistent` in this scope
    if let ((x, y if x) | (x if y, y)) = (true, true) { x && y; }
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `y` in this scope
    while let ((x, y if x) | (x if y, y)) = (true, true) { x && y; }
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `y` in this scope
    for ((x, y if x) | (x if y, y)) in [(true, true)] { x && y; }
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `y` in this scope

    (|(x if x), (y if y)| x && y)(true, true);
    (|(x if y), (y if x)| x && y)(true, true);
    //~^ ERROR cannot find value `x` in this scope
    //~| ERROR cannot find value `y` in this scope

    // FIXME(guard_patterns): mismatched bindings are not yet allowed
    match Some(0) {
        Some(x if x > 0) | None => {}
        //~^ ERROR variable `x` is not bound in all patterns
    }
}

/// Make sure shadowing is handled properly. In particular, if a pattern shadows an identifier,
/// a guard pattern's guard should still see the original binding if the shadowing binding isn't in
/// its subpattern.
fn test_shadowing(local: bool) -> u8 {
    match (0, 0) {
        // The `local` binding here shadows the `bool` definition, so we get a type error.
        //~v ERROR mismatched types
        local if local => 0,
        // The guards here should see the `bool` definition of `local`, not the new `u8` binding.
        // The body should see the new binding.
        (local, _ if local) => local,
        (_ if local, local) => local,
    }
}

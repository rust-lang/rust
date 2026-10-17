// Tests that `clippy::all` still works without a deprecation warning

//@require-annotations-for-level: WARN

#![deny(clippy::all)]

fn f() {
    "a".replace("a", "a");
    //~^ no_effect_replace
    //~| NOTE: implied by `#[deny(clippy::all)]`
}

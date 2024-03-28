// aux-build: const-unstable.rs
// compile-flags: -Zforce-unstable-if-unmarked
#![crate_type = "lib"]
extern crate const_unstable;

// Check that crates build with `-Zforce-unstable-if-unmarked` can't call
// const-unstable functions, despite their functions sometimes being considerd
// unstable.
//
// See https://github.com/rust-lang/rust/pull/118427#discussion_r1409914941 for
// more context.

pub const fn identity(x: i32) -> i32 {
    const_unstable::identity(x)
    //~^ ERROR `const_unstable::identity` is not yet stable as a const fn
}

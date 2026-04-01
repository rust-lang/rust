#![crate_type = "rlib"]
//@ edition: 2024
//@ check-pass

// Using `offset_of` in the RHS of a let-else statement should not produce
// malformed spans or a blank diagnostic snippet.
//
// Regression test for <https://github.com/rust-lang/rust/pull/152284>.

fn init_to_offset_of() {
    use std::mem::offset_of;
    struct Foo { field: u32 }

    if let x = offset_of!(Foo, field) {}
    //~^ WARN irrefutable `if let` pattern

    let x = offset_of!(Foo, field) else { return; };
    //~^ WARN irrefutable `let...else` pattern
}

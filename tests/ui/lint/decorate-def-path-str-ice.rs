// Checks that compiling this file with
// `-Dunused_must_use -Awarnings --cap-lints=warn --crate-type=lib` does not ICE when emitting
// diagnostics.
// Issue: <https://github.com/rust-lang/rust/issues/121774>.

//@ compile-flags: -Dunused_must_use -Awarnings --cap-lints=warn --crate-type=lib
//@ check-pass

#[must_use]
fn f() {}

pub fn g() {
    f();
}

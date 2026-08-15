// Passing link-args with an unexpected space
// could result in the flag being parsed and receiving
// an unexpected, empty linker argument. This test
// ensures successful compilation even when a space is
// present.
// See https://github.com/rust-lang/rust/pull/10749

//@ ignore-cross-compile
//@ ignore-windows-gnu
// Reason: The space is parsed as an empty linker argument on windows-gnu.

use run_make_support::rustc;

fn main() {
    // Notice the space at the end of -lc, which emulates the output of pkg-config.
    rustc().arg("-Clink-args=-lc ").input("empty.rs").run();
}

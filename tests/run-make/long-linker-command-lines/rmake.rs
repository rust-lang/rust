// This is a test which attempts to blow out the system limit with how many
// arguments can be passed to a process. This'll successively call rustc with
// larger and larger argument lists in an attempt to find one that's way too
// big for the system at hand. This file itself is then used as a "linker" to
// detect when the process creation succeeds.
//
// Eventually we should see an argument that looks like `@` as we switch from
// passing literal arguments to passing everything in the file.
// See https://github.com/rust-lang/rust/issues/41190

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{run, rustc};

fn main() {
    rustc().input("foo.rs").arg("-g").opt().run();
    run("foo");
}

// This test ensures that the compiler is keeping static variables, even if not referenced
// by another part of the program, in the output object file.
//
// It comes from #39987 which implements this RFC for the #[used] attribute:
// https://rust-lang.github.io/rfcs/2386-used.html

//@ ignore-msvc

use run_make_support::{cmd, rustc};

fn main() {
    rustc().opt_level("3").emit("obj").input("used.rs").run();

    cmd("nm").arg("used.o").run().assert_stdout_contains("FOO");
}

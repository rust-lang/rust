// In this test, foo.rs is compiled twice with different hashes tied to its
// symbols thanks to the metadata flag. bar.rs then ensures that the memory locations
// of foo's symbols are different even though they came from the same original source code.
// This checks that the metadata flag is doing its job.
// See https://github.com/rust-lang/rust/issues/14471

//@ ignore-cross-compile

use run_make_support::{run, rust_lib_name, rustc};

fn main() {
    rustc().input("foo.rs").metadata("a").extra_filename("-a").run();
    rustc().input("foo.rs").metadata("b").extra_filename("-b").run();
    rustc()
        .input("bar.rs")
        .extern_("foo1", rust_lib_name("foo-a"))
        .extern_("foo2", rust_lib_name("foo-b"))
        .run();
    run("bar");
}

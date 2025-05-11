// target-cpu is a codegen flag that generates code for the processor of the host machine
// running the compilation. This test is a sanity test that this flag does not cause any
// warnings when used, and that binaries produced by it can also be successfully executed.
// See https://github.com/rust-lang/rust/pull/23238

//@ ignore-cross-compile target-cpu=native doesn't work well when cross compiling

use run_make_support::{run, rustc};

fn main() {
    let out = rustc().input("foo.rs").arg("-Ctarget-cpu=native").run().stderr_utf8();
    run("foo");
    // There should be zero warnings emitted - the bug would cause "unknown CPU `native`"
    // to be printed out.
    assert!(out.is_empty());
}

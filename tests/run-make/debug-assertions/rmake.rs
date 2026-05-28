// debug.rs contains some "debug assertion" statements which
// should only be enabled in either non-optimized builds or when
// `-C debug-assertions` is set to yes. These debug assertions
// are guaranteed to fail, so this test checks that the run command
// fails where debug assertions should be activated, and succeeds where
// debug assertions should be disabled.
// See https://github.com/rust-lang/rust/pull/22980

//@ ignore-cross-compile
//@ needs-unwind

use run_make_support::{run, run_fail, rustc};

fn main() {
    rustc().input("debug.rs").arg("-Cdebug-assertions=no").run();
    run("debug");
    rustc().input("debug.rs").opt_level("0").run();
    run_fail("debug");
    rustc().input("debug.rs").opt_level("1").run();
    run("debug");
    rustc().input("debug.rs").opt_level("2").run();
    run("debug");
    rustc().input("debug.rs").opt_level("3").run();
    run("debug");
    rustc().input("debug.rs").opt_level("s").run();
    run("debug");
    rustc().input("debug.rs").opt_level("z").run();
    run("debug");
    rustc().input("debug.rs").opt().run();
    run("debug");
    rustc().input("debug.rs").run();
    run_fail("debug");
    rustc().input("debug.rs").opt().arg("-Cdebug-assertions=yes").run();
    run_fail("debug");
    rustc().input("debug.rs").opt_level("1").arg("-Cdebug-assertions=yes").run();
    run_fail("debug");
}

// When native libraries are passed to the linker, there used to be an annoyance
// where multiple instances of the same library in a row would cause duplication in
// outputs. This has been fixed, and this test checks that it stays fixed.
// With the --cfg flag, -ltestb gets added to the output, breaking up the chain of -ltesta.
// Without the --cfg flag, there should be a single -ltesta, no more, no less.
// See https://github.com/rust-lang/rust/pull/84794

//@ ignore-msvc

use run_make_support::rustc;

fn main() {
    rustc().input("depa.rs").run();
    rustc().input("depb.rs").run();
    rustc().input("depc.rs").run();
    let output = rustc().input("empty.rs").cfg("bar").run_fail();
    output.assert_stderr_contains(r#""-ltesta" "-ltestb" "-ltesta""#);
    let output = rustc().input("empty.rs").run_fail();
    output.assert_stderr_contains(r#""-ltesta""#);
    let output = rustc().input("empty.rs").run_fail();
    output.assert_stderr_not_contains(r#""-ltestb""#);
    let output = rustc().input("empty.rs").run_fail();
    output.assert_stderr_not_contains(r#""-ltesta" "-ltesta" "-ltesta""#);
}

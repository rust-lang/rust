//@ needs-target-std
//@ ignore-musl (not passed consecutively)
// When native libraries are passed to the linker, there used to be an annoyance
// where multiple instances of the same library in a row would cause duplication in
// outputs. This has been fixed, and this test checks that it stays fixed.
// With the --cfg flag, -ltestb gets added to the output, breaking up the chain of -ltesta.
// Without the --cfg flag, there should be a single -ltesta, no more, no less.
// See https://github.com/rust-lang/rust/pull/84794

use std::fmt::Write;

use run_make_support::{is_msvc, rustc, target};

fn main() {
    rustc().input("depa.rs").run();
    rustc().input("depb.rs").run();
    rustc().input("depc.rs").run();

    let output = rustc().input("empty.rs").cfg("bar").arg("--print=link-args").run_fail();
    output.assert_stdout_contains(needle_from_libs(&["testa", "testb", "testa"]));

    let output = rustc().input("empty.rs").arg("--print=link-args").run_fail();
    output.assert_stdout_contains(needle_from_libs(&["testa"]));
    output.assert_stdout_not_contains(needle_from_libs(&["testb"]));
    output.assert_stdout_not_contains(needle_from_libs(&["testa", "testa", "testa"]));
    // Adjacent identical native libraries are no longer deduplicated if
    // they come from different crates (https://github.com/rust-lang/rust/pull/103311)
    // so the following will fail:
    //output.assert_stderr_not_contains(needle_from_libs(&["testa", "testa"]));
}

fn needle_from_libs(libs: &[&str]) -> String {
    let mut needle = String::new();
    for lib in libs {
        if is_msvc() {
            needle.write_fmt(format_args!(r#""{lib}.lib" "#)).unwrap();
        } else if target().contains("wasm") {
            needle.write_fmt(format_args!(r#""-l" "{lib}" "#)).unwrap();
        } else {
            needle.write_fmt(format_args!(r#""-l{lib}" "#)).unwrap();
        }
    }
    needle.pop(); // remove trailing space
    needle
}

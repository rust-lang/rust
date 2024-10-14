// This test revolves around the rustc flag -Z profile, which should
// generate a .gcno file (initial profiling information) as well
// as a .gcda file (branch counters). The path where these are emitted
// should also be configurable with -Z profile-emit. This test checks
// that the files are produced, and then that the latter flag is respected.
// See https://github.com/rust-lang/rust/pull/42433

//@ ignore-cross-compile
//@ needs-profiler-runtime

use run_make_support::{path, run, rustc};

fn main() {
    rustc().arg("-g").arg("-Zprofile").input("test.rs").run();
    run("test");
    assert!(path("test.gcno").exists(), "no .gcno file");
    assert!(path("test.gcda").exists(), "no .gcda file");
    rustc().arg("-g").arg("-Zprofile").arg("-Zprofile-emit=abc/abc.gcda").input("test.rs").run();
    run("test");
    assert!(path("abc/abc.gcda").exists(), "gcda file not emitted to defined path");
}

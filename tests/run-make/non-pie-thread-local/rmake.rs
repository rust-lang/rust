// It was once required to use a position-independent executable (PIE)
// in order to use the thread_local! macro, or some symbols would contain
// a NULL address. This was fixed, and this test checks a non-PIE, then a PIE
// build to see if this bug makes a resurgence.
// See https://github.com/rust-lang/rust/pull/24448

//@ ignore-cross-compile
//@ only-linux

use run_make_support::{cc, cwd, run, rustc};

fn main() {
    rustc().input("foo.rs").run();
    cc().input("foo.c")
        .arg("-lfoo")
        .library_search_path(cwd())
        .arg("-Wl,--gc-sections")
        .arg("-lpthread")
        .arg("-ldl")
        .out_exe("foo")
        .run();
    run("foo");
    cc().input("foo.c")
        .arg("-lfoo")
        .library_search_path(cwd())
        .arg("-Wl,--gc-sections")
        .arg("-lpthread")
        .arg("-ldl")
        .arg("-pie")
        .arg("-fPIC")
        .out_exe("foo")
        .run();
    run("foo");
}

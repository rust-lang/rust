// Generation of position-independent code (PIC) can be altered
// through use of the -C relocation-model rustc flag. This test
// uses varied values with this flag and checks that compilation
// succeeds.
// See https://github.com/rust-lang/rust/pull/13340

//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    // FIXME(Oneirical): This first one will likely fail on MSVC due to #28026.
    // Remove this after try-job
    rustc().arg("-Crelocation-model=static").input("foo.rs").run();
    run("foo");
    rustc().arg("-Crelocation-model=dynamic-no-pic").input("foo.rs").run();
    run("foo");
    rustc().arg("-Crelocation-model=default").input("foo.rs").run();
    run("foo");
    rustc()
        .arg("-Crelocation-model=dynamic-no-pic")
        .crate_type("dylib")
        .emit("link,obj")
        .input("foo.rs")
        .run();
}

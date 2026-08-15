// When building a binary instrumented with mcount, verify the
// binary is linked with the correct crt to enable profiling.
//
//@ only-gnu
//@ ignore-cross-compile

use run_make_support::{path, run, rustc};

fn main() {
    // Compile instrumentation enabled binary, and verify -pg is passed
    let link_args =
        rustc().input("main.rs").arg("-Zinstrument-mcount").print("link-args").run().stdout_utf8();
    assert!(link_args.contains("\"-pg\""));

    // Run it, and verify gmon.out is created
    assert!(!path("gmon.out").exists());
    run("main");
    assert!(path("gmon.out").exists());
}

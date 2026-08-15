//! Test that linking works under an environment similar to what Xcode sets up.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/80817.

//@ only-apple

use run_make_support::{cmd, rustc, target};

fn main() {
    // Fetch toolchain `/usr/bin` directory. Usually:
    // /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
    let clang_bin = cmd("xcrun").arg("--find").arg("clang").run().stdout_utf8();
    let toolchain_bin = clang_bin.trim().strip_suffix("/clang").unwrap();

    // Put toolchain directory at the front of PATH.
    let path = format!("{}:{}", toolchain_bin, std::env::var("PATH").unwrap());

    // Check that compiling and linking still works.
    //
    // Removing `SDKROOT` is necessary for the test to excercise what we want, since bootstrap runs
    // under `/usr/bin/python3`, which will set SDKROOT for us.
    rustc().target(target()).env_remove("SDKROOT").env("PATH", &path).input("foo.rs").run();

    // Also check linking directly with the system linker.
    rustc()
        .target(target())
        .env_remove("SDKROOT")
        .env("PATH", &path)
        .input("foo.rs")
        .arg("-Clinker-flavor=ld")
        .run();
}

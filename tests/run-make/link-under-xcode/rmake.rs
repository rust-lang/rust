//! Test that linking works under an environment similar to what Xcode sets up.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/80817.

//@ only-apple

use run_make_support::{cmd, rustc, target};

fn main() {
    // Fetch toolchain `/usr/bin` directory. Usually:
    // /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
    let cc_bin = cmd("xcrun").arg("--find").arg("cc").run().stdout_utf8();
    let toolchain_bin = cc_bin.trim().strip_suffix("/cc").unwrap();

    // Put toolchain directory at the front of PATH.
    let path = format!("{}:{}", toolchain_bin, std::env::var("PATH").unwrap());

    // Check that compiling and linking still works.
    //
    // Removing `SDKROOT` is necessary for the test to excercise what we want, since bootstrap runs
    // under `/usr/bin/python3`, which will set SDKROOT for us.
    rustc().target(target()).env_remove("SDKROOT").env("PATH", &path).input("foo.rs").run();

    // Also check with ld64.
    rustc()
        .target(target())
        .env_remove("SDKROOT")
        .env("PATH", &path)
        .arg("-Clinker-flavor=ld")
        .input("foo.rs")
        .run();
}

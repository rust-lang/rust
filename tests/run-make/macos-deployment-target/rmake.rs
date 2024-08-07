// Check that a set deployment target actually makes it to the linker.
// This is important since its a compatibility hazard. The linker will
// generate load commands differently based on what minimum OS it can assume.
// See https://github.com/rust-lang/rust/pull/105123

//@ only-macos
// Reason: this test exercises an OSX-specific issue

use run_make_support::{cmd, rustc};

fn main() {
    rustc()
        .env("MACOSX_DEPLOYMENT_TARGET", "10.13")
        .input("with_deployment_target.rs")
        .output("with_deployment_target.dylib")
        .run();
    let pattern = if cmd("uname").arg("-m").run().stdout_utf8().contains("arm64") {
        "minos 11.0"
    } else {
        "version 10.13"
    };
    // NOTE: The check is for either the x86_64 minimum OR the aarch64 minimum
    // (M1 starts at macOS 11). They also use different load commands, so we let that change with
    // each too. The aarch64 check isn't as robust as the x86 one, but testing both seems unneeded.
    cmd("vtool")
        .arg("-show-build")
        .arg("with_deployment_target.dylib")
        .run()
        .assert_stdout_contains("pattern");
}

//! Test codegen when setting SDK version on Apple platforms.
//!
//! This is important since its a compatibility hazard. The linker will
//! generate load commands differently based on what minimum OS it can assume.
//!
//! See https://github.com/rust-lang/rust/issues/129432.

//@ only-apple

use run_make_support::{apple_os, cmd, run_in_tmpdir, rustc, target};

/// Run vtool to check the `sdk` field in LC_BUILD_VERSION.
///
/// On lower deployment targets, LC_VERSION_MIN_MACOSX, LC_VERSION_MIN_IPHONEOS and similar
/// are used instead of LC_BUILD_VERSION, but both name the relevant variable `sdk`.
#[track_caller]
fn has_sdk_version(file: &str, version: &str) {
    cmd("vtool")
        .arg("-show-build")
        .arg(file)
        .run()
        .assert_stdout_contains(format!("sdk {version}"));
}

fn main() {
    // Fetch rustc's inferred deployment target.
    let current_deployment_target = rustc().print("deployment-target").run().stdout_utf8();
    let current_deployment_target = current_deployment_target.split('=').last().unwrap().trim();

    // Fetch current SDK version via. xcrun.
    //
    // Assumes a standard Xcode distribution, where e.g. the macOS SDK's Mac Catalyst
    // and the iPhone Simulator version is the same as for the iPhone SDK.
    let sdk_name = match apple_os() {
        "macos" => "macosx",
        "ios" => "iphoneos",
        "watchos" => "watchos",
        "tvos" => "appletvos",
        "visionos" => "xros",
        _ => unreachable!(),
    };
    let current_sdk_version =
        cmd("xcrun").arg("--show-sdk-version").arg("--sdk").arg(sdk_name).run().stdout_utf8();
    let current_sdk_version = current_sdk_version.trim();

    // Check the SDK version in the object file produced by the codegen backend.
    rustc().crate_type("lib").emit("obj").input("foo.rs").output("foo.o").run();
    // Set to 0, which means not set or "n/a".
    has_sdk_version("foo.o", "n/a");

    // Check the SDK version in the .rmeta file, as set in `create_object_file`.
    //
    // This is just to ensure that we don't set some odd version in `create_object_file`,
    // if the rmeta file is packed in a different way in the future, this can safely be removed.
    rustc().crate_type("rlib").input("foo.rs").output("libfoo.rlib").run();
    // Extra .rmeta file (which is encoded as an object file).
    cmd("ar").arg("-x").arg("libfoo.rlib").arg("lib.rmeta").run();
    has_sdk_version("lib.rmeta", "n/a");

    // Test that version makes it to the linker.
    for (crate_type, file_ext) in [("bin", ""), ("dylib", ".dylib")] {
        // Non-simulator watchOS targets don't support dynamic linking,
        // for simplicity we disable the test on all watchOS targets.
        if crate_type == "dylib" && apple_os() == "watchos" {
            continue;
        }

        // Test with clang
        let file_name = format!("foo_cc{file_ext}");
        rustc()
            .crate_type("bin")
            .arg("-Clinker-flavor=gcc")
            .input("foo.rs")
            .output(&file_name)
            .run();
        has_sdk_version(&file_name, current_sdk_version);

        // Test with ld64
        let file_name = format!("foo_ld{file_ext}");
        rustc()
            .crate_type("bin")
            .arg("-Clinker-flavor=ld")
            .input("foo.rs")
            .output(&file_name)
            .run();
        // FIXME(madsmtm): This uses the current deployment target
        // instead of the current SDK version like Clang does.
        // https://github.com/rust-lang/rust/issues/129432
        has_sdk_version(&file_name, current_deployment_target);
    }
}

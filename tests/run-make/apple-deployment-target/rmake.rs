//! Test codegen when setting deployment targets on Apple platforms.
//!
//! This is important since its a compatibility hazard. The linker will
//! generate load commands differently based on what minimum OS it can assume.
//!
//! See https://github.com/rust-lang/rust/pull/105123.

//@ only-apple

use std::collections::HashSet;

use run_make_support::{
    apple_os, cmd, has_extension, path, regex, run_in_tmpdir, rustc, shallow_find_files, target,
};

/// Run vtool to check the `minos` field in LC_BUILD_VERSION.
///
/// On lower deployment targets, LC_VERSION_MIN_MACOSX, LC_VERSION_MIN_IPHONEOS and similar
/// are used instead of LC_BUILD_VERSION - these have a `version` field, so also check that.
#[track_caller]
fn minos(file: &str, version: &str) {
    cmd("vtool")
        .arg("-show-build")
        .arg(file)
        .run()
        .assert_stdout_contains_regex(format!("(minos|version) {version}"));
}

fn main() {
    // These versions should generally be higher than the default versions
    let (example_version, higher_example_version) = match apple_os() {
        "macos" => ("12.0", "13.0"),
        // armv7s-apple-ios and i386-apple-ios only supports iOS 10.0
        "ios" if target() == "armv7s-apple-ios" || target() == "i386-apple-ios" => ("10.0", "10.0"),
        "ios" => ("15.0", "16.0"),
        "watchos" => ("7.0", "9.0"),
        "tvos" => ("14.0", "15.0"),
        "visionos" => ("1.1", "1.2"),
        _ => unreachable!(),
    };

    // Remove env vars to get `rustc`'s default
    let output = rustc()
        .env_remove("MACOSX_DEPLOYMENT_TARGET")
        .env_remove("IPHONEOS_DEPLOYMENT_TARGET")
        .env_remove("WATCHOS_DEPLOYMENT_TARGET")
        .env_remove("TVOS_DEPLOYMENT_TARGET")
        .env_remove("XROS_DEPLOYMENT_TARGET")
        .print("deployment-target")
        .run()
        .stdout_utf8();
    let (env_var, default_version) = output.split_once('=').unwrap();
    let env_var = env_var.trim();
    let default_version = default_version.trim();

    // Test that version makes it to the object file.
    run_in_tmpdir(|| {
        let rustc = || {
            let mut rustc = rustc();
            rustc.crate_type("lib");
            rustc.emit("obj");
            rustc.input("foo.rs");
            rustc.output("foo.o");
            rustc
        };

        rustc().env(env_var, example_version).run();
        minos("foo.o", example_version);

        rustc().env_remove(env_var).run();
        minos("foo.o", default_version);
    });

    // Test that version makes it to the linker when linking dylibs.
    run_in_tmpdir(|| {
        // Certain watchOS targets don't support dynamic linking, so we disable the test on those.
        if apple_os() == "watchos" {
            return;
        }

        let rustc = || {
            let mut rustc = rustc();
            rustc.crate_type("dylib");
            rustc.input("foo.rs");
            rustc.output("libfoo.dylib");
            rustc
        };

        rustc().env(env_var, example_version).run();
        minos("libfoo.dylib", example_version);

        rustc().env_remove(env_var).run();
        minos("libfoo.dylib", default_version);

        // Test with ld64 instead

        rustc().arg("-Clinker-flavor=ld").env(env_var, example_version).run();
        minos("libfoo.dylib", example_version);

        rustc().arg("-Clinker-flavor=ld").env_remove(env_var).run();
        minos("libfoo.dylib", default_version);
    });

    // Test that version makes it to the linker when linking executables.
    run_in_tmpdir(|| {
        let rustc = || {
            let mut rustc = rustc();
            rustc.crate_type("bin");
            rustc.input("foo.rs");
            rustc.output("foo");
            rustc
        };

        // FIXME(madsmtm): Xcode's version of Clang seems to require a minimum
        // version of 9.0 on aarch64-apple-watchos for some reason? Which is
        // odd, because the first Aarch64 watch was Apple Watch Series 4,
        // which runs on as low as watchOS 5.0.
        //
        // You can see Clang's behaviour by running:
        // ```
        // echo "int main() { return 0; }" > main.c
        // xcrun --sdk watchos clang --target=aarch64-apple-watchos main.c
        // vtool -show a.out
        // ```
        if target() != "aarch64-apple-watchos" {
            rustc().env(env_var, example_version).run();
            minos("foo", example_version);

            rustc().env_remove(env_var).run();
            minos("foo", default_version);
        }

        // Test with ld64 instead

        rustc().arg("-Clinker-flavor=ld").env(env_var, example_version).run();
        minos("foo", example_version);

        rustc().arg("-Clinker-flavor=ld").env_remove(env_var).run();
        minos("foo", default_version);
    });

    // Test that changing the deployment target busts the incremental cache.
    run_in_tmpdir(|| {
        let rustc = || {
            let mut rustc = rustc();
            rustc.incremental("incremental");
            rustc.crate_type("lib");
            rustc.emit("obj");
            rustc.input("foo.rs");
            rustc.output("foo.o");
            rustc
        };

        // FIXME(madsmtm): Incremental cache is not yet busted
        // https://github.com/rust-lang/rust/issues/118204
        let higher_example_version = example_version;
        let default_version = example_version;

        rustc().env(env_var, example_version).run();
        minos("foo.o", example_version);

        rustc().env(env_var, higher_example_version).run();
        minos("foo.o", higher_example_version);

        rustc().env_remove(env_var).run();
        minos("foo.o", default_version);
    });

    // Test that all binaries in rlibs produced by `rustc` have the same version.
    // Regression test for https://github.com/rust-lang/rust/issues/128419.
    let sysroot = rustc().print("sysroot").run().stdout_utf8();
    let target_sysroot = path(sysroot.trim()).join("lib/rustlib").join(target()).join("lib");
    let rlibs = shallow_find_files(&target_sysroot, |path| has_extension(path, "rlib"));

    let output = cmd("otool").arg("-l").args(rlibs).run().stdout_utf8();
    let re = regex::Regex::new(r"(minos|version) ([0-9.]*)").unwrap();
    let mut versions = HashSet::new();
    for (_, [_, version]) in re.captures_iter(&output).map(|c| c.extract()) {
        versions.insert(version);
    }
    // FIXME(madsmtm): See above for aarch64-apple-watchos.
    if versions.len() != 1 && target() != "aarch64-apple-watchos" {
        panic!("std rlibs contained multiple different deployment target versions: {versions:?}");
    }
}

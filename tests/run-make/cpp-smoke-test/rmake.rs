//! `backtrace`'s `cpp_smoke_test` ported to rust-lang/rust.
//!
//! A basic smoke test that exercises `backtrace` to see if it can resolve basic C++ templated +
//! trampolined symbol names.

//@ ignore-cross-compile (binary needs to run)
//@ only-nightly

//@ ignore-windows-msvc (test fails due to backtrace symbol mismatches)
// FIXME: on MSVC, at `-O1`, there are no symbols available. At `-O0`, the test fails looking like:
// ```
// actual names = [
//     "space::templated_trampoline<void (__cdecl*)(void)>",
// ]
// expected names = [
//     "void space::templated_trampoline<void (*)()>(void (*)())",
//     "cpp_trampoline",
// ]
// ```

use run_make_support::rustc::sysroot;
use run_make_support::{
    build_native_static_lib_cxx_optimized, cargo, crate_cc, cwd, path, rfs, run, rustc,
    source_root, target,
};

fn main() {
    let target_dir = path("target");
    let src_root = source_root();
    let backtrace_submodule = src_root.join("library").join("backtrace");
    let backtrace_toml = backtrace_submodule.join("Cargo.toml");

    // Build the `backtrace` package (the `library/backtrace` submodule to make sure we exercise the
    // same `backtrace` as shipped with std).
    cargo()
        // NOTE: needed to skip trying to link in `windows.0.52.0.lib` which is pre-built but not
        // available in *this* scenario.
        .env("RUSTFLAGS", "--cfg=windows_raw_dylib")
        .arg("build")
        .args(&["--manifest-path", &backtrace_toml.to_str().unwrap()])
        .args(&["--target", &target()])
        .arg("--features=cpp_demangle")
        .env("CARGO_TARGET_DIR", &target_dir)
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .run();

    let rlibs_path = target_dir.join(target()).join("debug").join("deps");

    // FIXME: this test is *really* fragile. Even on `x86_64-unknown-linux-gnu`, this fails if a
    // different opt-level is passed. On `-O2` this test fails due to no symbols found. On `-O0`
    // this test fails because it's missing one of the expected symbols.
    build_native_static_lib_cxx_optimized("trampoline", "-O1");

    rustc()
        .input("cpp_smoke_test.rs")
        .library_search_path(&rlibs_path)
        .library_search_path(cwd())
        .debuginfo("2")
        .arg("-ltrampoline")
        .run();

    run("cpp_smoke_test");
}

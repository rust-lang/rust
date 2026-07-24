// This test makes sure that the injected `compiler_builtins` can be loaded both from
// `-L crate=` and `-L dependency=` paths (apart from sysroot).
//
// The need for `-L dependency=` paths comes from RFC 3874 (build-std=always).
//
// Note: We have two possible `compiler_builtins` crates: the built one and the one from
// the sysroot. Sysroot lookup is disabled via the `--sysroot=` override to verify
// that we can load the correct one.

use run_make_support::{path, rfs, rust_lib_name, rustc};

fn main() {
    rfs::create_dir("compiler_builtins");

    // Compile `core`.
    rustc()
        .input("core.rs")
        .arg("-Zunstable-options")
        .panic("immediate-abort")
        .arg("--edition=2024")
        .sysroot("./no_exists")
        .run();

    // Compile `compiler_builtins` into a separate directory to prevent it from being
    // found via `-L .`.
    rustc()
        .input("compiler_builtins.rs")
        .arg("-Zunstable-options")
        .panic("immediate-abort")
        .arg("--edition=2024")
        .out_dir("compiler_builtins")
        .sysroot("./no_exists")
        .run();

    // Compile the final artifact. `compiler_builtins` cannot be located without the
    // `-L{crate/dependency}=` option.
    rustc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .panic("immediate-abort")
        .arg("--edition=2024")
        .sysroot("./no_exists")
        .run_fail()
        .assert_stderr_contains("can't find crate for `compiler_builtins`");

    // Compile the final artifact. `compiler_builtins` can be located via
    // `-Lcrate=` paths.
    rustc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .panic("immediate-abort")
        .arg("--edition=2024")
        .sysroot("./sysroot")
        .library_search_path(format!("crate={}", path("compiler_builtins").display()))
        .run();

    // Compile the final artifact. The `compiler_builtins` can be located via
    // `-Ldependency=` paths.
    rustc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .panic("immediate-abort")
        .arg("--edition=2024")
        .sysroot("./no_exists")
        .library_search_path(format!("dependency={}", path("compiler_builtins").display()))
        .run_fail()
        .assert_stderr_contains("can't find crate for `compiler_builtins`");
}

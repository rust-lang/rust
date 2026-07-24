// This test makes sure that the injected panic runtime can be loaded both from
// `-L crate=` and `-L dependency=` paths.
//
// The need for `-L dependency=` paths comes from RFC 3874 (build-std=always).
//
// Note: We have two possible panic runtime crates: the built one and the one from
// the sysroot. Sysroot lookup is disabled via the `--sysroot=` override to verify
// that we can load the correct one.

use run_make_support::{path, rfs, rust_lib_name, rustc};

fn main() {
    rfs::create_dir("panic_abort");

    // Compile `core`.
    rustc().input("core.rs").panic("abort").sysroot("./no_exists").run();

    // Compile `panic_abort` into a separate directory to prevent it from being
    // found via `-L .`.
    rustc()
        .input("panic_abort.rs")
        .panic("abort")
        .out_dir("panic_abort")
        .sysroot("./no_exists")
        .run();

    // Compile `std`.
    rustc()
        .input("std.rs")
        .extern_("panic_abort", &path("panic_abort").join(rust_lib_name("panic_abort")))
        .panic("abort")
        .sysroot("./no_exists")
        .run();

    // Compile the final artifact. The panic runtime cannot be located without the
    // `-L{crate/dependency}=` option.
    rustc()
        .input("lib.rs")
        .arg("-Cpanic=abort")
        .sysroot("./no_exists")
        .run_fail()
        .assert_stderr_contains("can't find crate for `panic_abort`");

    // Compile the final artifact. The panic runtime can be located via
    // `-Lcrate=` paths.
    rustc()
        .input("lib.rs")
        .arg("-Cpanic=abort")
        .sysroot("./no_exists")
        .library_search_path(format!("crate={}", path("panic_abort").display()))
        .run();

    // Compile the final artifact. The panic runtime can be located via
    // `-Ldependency=` paths.
    rustc()
        .input("lib.rs")
        .arg("-Cpanic=abort")
        .sysroot("./no_exists")
        .library_search_path(format!("dependency={}", path("panic_abort").display()))
        .run_fail()
        .assert_stderr_contains("can't find crate for `panic_abort`");
}

// This test makes sure that the injected panic runtime can be loaded from
// `-L dependency=` paths as per RFC 3874 (build-std=always).
//
// Note: We have two possible panic runtime crates: the built one and the one from
// the sysroot. Sysroot lookup is disabled via the `--sysroot=` override to verify
// that we can load the correct one.
//
// `--emit=llvm-ir` is used to avoid running the linker.

use run_make_support::{path, rfs, rust_lib_name, rustc};

fn main() {
    rfs::create_dir("panic_abort");

    // Compile `core`.
    rustc().input("core.rs").panic("abort").sysroot("./no_exists").run();

    // Compile `panic_abort` into a separate directory to prevent it from being
    // found via `-L .`
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
    // `-Ldependency=` option.
    rustc()
        .input("lib.rs")
        .arg("-Cpanic=abort")
        .sysroot("./no_exists")
        .emit("llvm-ir")
        .run_fail()
        .assert_stderr_contains("can't find crate for `panic_abort`");

    // Compile the final artifact. The panic runtime cannot be located via
    // `-Lcrate=` paths (This means that the panic runtime is not direct
    // dependency).
    rustc()
        .input("lib.rs")
        .arg("-Cpanic=abort")
        .sysroot("./no_exists")
        .library_search_path(format!("crate={}", path("panic_abort").display()))
        .emit("llvm-ir")
        .run_fail()
        .assert_stderr_contains("can't find crate for `panic_abort`");

    // Compile the final artifact. The panic runtime can be located via
    // `-Ldependency=` paths.
    rustc()
        .input("lib.rs")
        .arg("-Cpanic=abort")
        .sysroot("./no_exists")
        .library_search_path(format!("dependency={}", path("panic_abort").display()))
        .emit("llvm-ir")
        .run();
}

// Test compilation flow using custom pauth-enabled toolchain and signing extern "C" function
// pointers used from within rust. The test assumes that pointer-authentication-enabled `clang` is
// available on the path. In this test rust is the driver - providing the data and the comparison
// function; while c - provides the implementation of quicksort algorithm and is the user of  the
// data and comparator.

//@ only-aarch64-unknown-linux-pauthtest

use run_make_support::{cc, env_var, rfs, run, run_fail, rustc};

fn main() {
    // Use CC and CC_DEFAULT_FLAGS env variables to set up linker for rustc. This results in the
    // same command as cc(). The CC env variable corresponds to cc field in the config toml file.
    // This field is required to point to a clang family compiler on aarch64-unknown-linux-pauthtest
    // target.
    let rust_lib_name = "rust_quicksort";
    rustc()
        .target("aarch64-unknown-linux-pauthtest")
        .crate_type("cdylib")
        .input("quicksort.rs")
        .linker(&env_var("CC"))
        .link_args(&env_var("CC_DEFAULT_FLAGS"))
        .crate_name(rust_lib_name)
        .run();

    let exe_name = "main";
    cc().out_exe(exe_name)
        .input("main.c")
        .args(&[
            "-march=armv8.3-a+pauth",
            "-target",
            "aarch64-unknown-linux-pauthtest",
            &format!("-l{}", rust_lib_name),
        ])
        .library_search_path(".")
        .run();

    run(exe_name);

    rfs::remove_file(format!("{}{rust_lib_name}.{}", "lib", "so"));
    run_fail(exe_name);
}

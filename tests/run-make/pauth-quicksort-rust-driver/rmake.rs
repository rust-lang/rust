// Test compilation flow using custom pauth-enabled toolchain and signing extern "C" function
// pointers used from within rust. The test assumes that pointer-authentication-enabled `clang` is
// available on the path.
// In this test rust is the driver - providing the data and the comparison function; while c -
// provides the implementation of quicksort algorithm and is the user of  the data and comparator.

//@ only-aarch64-unknown-linux-pauthtest

use run_make_support::{cc, env_var, rfs, run, run_fail, rustc};

fn main() {
    let input = "quicksort";
    let input_name = format!("{input}.c");
    let lib_name = format!("{}{input}.{}", "lib", "so");
    cc().out_exe(&lib_name)
        .input(&input_name)
        .args(&["-target", "aarch64-unknown-linux-pauthtest", "-march=armv8.3-a+pauth", "-shared"])
        .run();

    // Use CC and CC_DEFAULT_FLAGS env variables to set up linker for rustc. This results in the
    // same command as cc(). The CC env variable corresponds to cc field in the config toml file.
    // This field is required to point to a clang family compiler on aarch64-unknown-linux-pauthtest
    // target.
    rustc()
        .target("aarch64-unknown-linux-pauthtest")
        .input("main.rs")
        .linker(&env_var("CC"))
        .link_args(&env_var("CC_DEFAULT_FLAGS"))
        .run();
    run("main");

    rfs::remove_file(&lib_name);
    run_fail("main");
}

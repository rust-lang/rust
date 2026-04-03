// Test compilation flow using custom pauth-enabled toolchain and signing extern "C" function
// pointers used from within rust. The test assumes that pauthtest-enabled `clang` is available on
// the path.
// In this test rust is the driver - providing the data and the comparison function; while c -
// provides the implementation of quicksort algorithm and is the user of  the data and comparator.

//@ only-aarch64-unknown-linux-pauthtest

use run_make_support::{cc, rfs, run, run_fail, rustc};

fn main() {
    unsafe {
        std::env::set_var("CC", "clang");
    }

    let pauthtest_sysroot = std::env::var("PAUTHTEST_SYSROOT").unwrap_or_default();
    let input = "quicksort";
    let input_name = format!("{input}.c");
    let lib_name = format!("{}{input}.{}", "lib", "so");
    cc().out_exe(&lib_name)
        .input(&input_name)
        .args(&[
            &format!("--sysroot={}", pauthtest_sysroot),
            "-lc",
            "-nostdlib",
            "-target",
            "aarch64-linux-pauthtest",
            "-fPIC",
            "-shared",
        ])
        .run();

    rustc().target("aarch64-unknown-linux-pauthtest").input("main.rs").run();
    run("main");

    rfs::remove_file(&lib_name);
    run_fail("main");
}

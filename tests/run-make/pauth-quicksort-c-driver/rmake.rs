// Test compilation flow using custom pauth-enabled toolchain and signing extern "C" function
// pointers used from within rust. Note that in order for the test to work the location of the
// toolchain's sysroot has to be provided via env variable (`PAUTHTEST_SYSROOT`). The test assumes
// that pauthtest-enabled `clang` is available on the path.
// In this test rust is the driver - providing the data and the comparison function; while c -
// provides the implementation of quicksort algorithm and is the user of  the data and comparator.

//@ only-aarch64-unknown-linux-pauthtest

use run_make_support::{cc, rfs, run, run_fail, rustc};

fn main() {
    unsafe {
        std::env::set_var("CC", "clang");
    }
    let pauthtest_sysroot = std::env::var("PAUTHTEST_SYSROOT").unwrap_or_default();
    let dynamic_linker = format!("-Wl,--dynamic-linker={}/usr/lib/libc.so", pauthtest_sysroot);
    let rpath = format!("-Wl,--rpath={}/usr/lib", pauthtest_sysroot);

    let rust_lib_name = "rust_quicksort";
    rustc()
        .target("aarch64-unknown-linux-pauthtest")
        .crate_type("cdylib")
        .input("quicksort.rs")
        .crate_name(rust_lib_name)
        .args(&[&dynamic_linker, &rpath])
        .run();

    let exe_name = "main";
    cc().out_exe(exe_name)
        .input("main.c")
        .args(&[
            "-march=armv8.3-a",
            "-lc",
            "-nostdlib",
            "-target",
            "aarch64-unknown-linux-pauthtest",
            "-fuse-ld=lld".into(),
            &format!("--sysroot={}", pauthtest_sysroot),
            "-I",
            &format!("{}/usr/include", pauthtest_sysroot),
            &format!("-Wl,--rpath={}/usr/lib", pauthtest_sysroot),
            &format!("-Wl,{}/usr/lib/crt1.o", pauthtest_sysroot),
            &format!("-Wl,{}/usr/lib/crti.o", pauthtest_sysroot),
            &format!("-Wl,{}/usr/lib/crtn.o", pauthtest_sysroot),
            "-L.",
            &format!("-l{}", rust_lib_name),
            &dynamic_linker,
            &rpath,
        ])
        .run();

    run(exe_name);

    rfs::remove_file(format!("{}{rust_lib_name}.{}", "lib", "so"));
    run_fail(exe_name);
}

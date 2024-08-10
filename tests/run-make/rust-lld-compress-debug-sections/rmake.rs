// Checks the `compress-debug-sections` option on rust-lld.

//@ needs-rust-lld
//@ needs-llvm-zstd
//@ only-linux
//@ ignore-cross-compile

// FIXME: This test isn't comprehensive and isn't covering all possible combinations.

use run_make_support::{llvm_readobj, run_in_tmpdir, rustc};

fn check_compression(compression: &str, to_find: &str) {
    run_in_tmpdir(|| {
        let out = rustc()
            .arg("-Zlinker-features=+lld")
            .arg("-Clink-self-contained=+linker")
            .arg("-Zunstable-options")
            .arg("-Cdebuginfo=full")
            .link_arg(&format!("-Wl,--compress-debug-sections={compression}"))
            .input("main.rs")
            .run();
        llvm_readobj().arg("-t").arg("main").run().assert_stdout_contains(to_find);
    });
}

fn main() {
    check_compression("zlib", "ZLIB");
    check_compression("zstd", "ZSTD");
}

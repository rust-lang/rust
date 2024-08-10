// Checks the always enabled `debuginfo-compression` option: zlib.

//@ only-linux
//@ ignore-cross-compile

use run_make_support::{llvm_readobj, run_in_tmpdir, rustc};

fn check_compression(compression: &str, to_find: &str) {
    run_in_tmpdir(|| {
        let out = rustc()
            .crate_name("foo")
            .crate_type("lib")
            .emit("obj")
            .arg("-Cdebuginfo=full")
            .arg(&format!("-Zdebuginfo-compression={compression}"))
            .input("foo.rs")
            .run();
        llvm_readobj().arg("-t").arg("foo.o").run().assert_stdout_contains(to_find);
    });
}

fn main() {
    check_compression("zlib", "ZLIB");
}

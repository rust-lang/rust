// Checks the `debuginfo-compression` option.

//@ only-linux
//@ ignore-cross-compile

// FIXME: This test isn't comprehensive and isn't covering all possible combinations.

use run_make_support::{assert_contains, llvm_readobj, run_in_tmpdir, rustc};

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
        let stderr = out.stderr_utf8();
        if stderr.is_empty() {
            llvm_readobj().arg("-t").arg("foo.o").run().assert_stdout_contains(to_find);
        } else {
            assert_contains(
                stderr,
                format!("unknown debuginfo compression algorithm {compression}"),
            );
        }
    });
}

fn main() {
    check_compression("zlib", "ZLIB");
    check_compression("zstd", "ZSTD");
}

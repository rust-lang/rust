// Test that LTO builds work correctly with -Z stable-crate-hash,
// including correct panic locations for cross-crate inlined functions.

//@ ignore-cross-compile

use run_make_support::{cmd, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .crate_name("rdr_lto_lib")
            .arg("-Zstable-crate-hash")
            .arg("-Clto=thin")
            .run();

        rustc()
            .input("main.rs")
            .crate_type("bin")
            .extern_("rdr_lto_lib", "librdr_lto_lib.rlib")
            .arg("-Zstable-crate-hash")
            .arg("-Clto=thin")
            .output("main")
            .run();

        cmd("./main").run();

        let output = cmd("./main").arg("panic").run_fail();
        let stderr = output.stderr_utf8();
        assert!(
            stderr.contains("lib.rs:12"),
            "inlined panic should show lib.rs:12, got:\n{stderr}"
        );
    });
}

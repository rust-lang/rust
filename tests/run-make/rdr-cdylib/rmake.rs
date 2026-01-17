// Test that cdylib builds work correctly with -Z separate-spans.

//@ ignore-cross-compile

use run_make_support::{dynamic_lib_name, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        rustc()
            .input("lib.rs")
            .crate_type("cdylib")
            .crate_name("rdr_cdylib")
            .arg("-Zseparate-spans")
            .run();

        let lib_name = dynamic_lib_name("rdr_cdylib");
        assert!(
            std::path::Path::new(&lib_name).exists(),
            "cdylib should be created: {lib_name}"
        );
    });
}

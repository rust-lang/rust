//@ needs-target-std
use run_make_support::{htmldocck, rustdoc};

fn main() {
    let out_dir = "rustdoc";

    rustdoc()
        .input("src/lib.rs")
        .crate_name("foobar")
        .crate_type("lib")
        // This is intentionally using `-o` option flag and not the `output()` method.
        .arg("-o")
        .arg(&out_dir)
        .run();

    htmldocck().arg(out_dir).arg("src/lib.rs").run();
}

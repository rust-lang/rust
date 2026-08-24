//@ needs-target-std

use run_make_support::{htmldocck, path, rustc, rustdoc};

fn main() {
    let out_dir = path("out");

    rustdoc()
        .input("quebec.rs")
        .crate_name("quebec")
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg("--enable-index-page")
        .run();

    htmldocck().arg(&out_dir).arg("quebec.rs").run();
}

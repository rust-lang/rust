//@ needs-target-std

use run_make_support::{htmldocck, path, rustc, rustdoc};

fn main() {
    let parts_out_dir = path("parts");
    let out_dir = path("out");

    rustdoc()
        .input("quebec.rs")
        .crate_name("quebec")
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg(format!("--write-doc-meta-dir={}", parts_out_dir.display()))
        .run();
    assert!(parts_out_dir.join("quebec.json").exists());

    rustdoc()
        .arg("-Zunstable-options")
        .out_dir(&out_dir)
        .arg(format!("--read-doc-meta-dir={}", parts_out_dir.display()))
        .arg("--enable-index-page")
        .run();

    htmldocck().arg(&out_dir).arg("quebec.rs").run();
}

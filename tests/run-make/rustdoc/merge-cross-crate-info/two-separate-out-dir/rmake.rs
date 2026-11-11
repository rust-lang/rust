//@ needs-target-std

use run_make_support::{cwd, htmldocck, path, rust_lib_name, rustc, rustdoc};

fn main() {
    let out_dir = path("out");
    let alt_out_dir = path("alt-out");

    rustc().input("foxtrot.rs").crate_name("foxtrot").crate_type("rlib").run();
    rustdoc()
        .input("foxtrot.rs")
        .crate_name("foxtrot")
        .out_dir(&alt_out_dir)
        .arg("-Zunstable-options")
        .arg("--write-doc-meta-dir=info/doc.parts/foxtrot")
        .run();

    rustdoc()
        .input("echo.rs")
        .crate_name("echo")
        .extern_("foxtrot", rust_lib_name("foxtrot"))
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg("--write-doc-meta-dir=info/doc.parts/echo")
        .run();

    rustdoc()
        .arg("-Zunstable-options")
        .out_dir(&out_dir)
        .arg("--read-doc-meta-dir=info/doc.parts/echo")
        .arg("--read-doc-meta-dir=info/doc.parts/foxtrot")
        .arg("--enable-index-page")
        .run();

    htmldocck().arg(&out_dir).arg("echo.rs").run();
}

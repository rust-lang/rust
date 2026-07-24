//@ needs-target-std

use run_make_support::{cwd, htmldocck, path, rfs, rust_lib_name, rustc, rustdoc};

fn main() {
    let merged_dir = path("merged");
    let parts_out_dir = path("parts");
    let lib_dir = path("lib");
    let out_dir = path("out");

    rustc().input("quebec.rs").crate_name("quebec").crate_type("rlib").run();
    rustdoc()
        .input("quebec.rs")
        .crate_name("quebec")
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg("--write-doc-meta-dir=parts-wrong")
        .run();

    rustc()
        .input("tango.rs")
        .crate_name("tango")
        .crate_type("rlib")
        .extern_("quebec", rust_lib_name("quebec"))
        .run();
    rustdoc()
        .input("tango.rs")
        .crate_name("tango")
        .extern_("quebec", rust_lib_name("quebec"))
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg("--write-doc-meta-dir=parts-wrong")
        .run();

    rustdoc()
        .input("sierra.rs")
        .crate_name("sierra")
        .extern_("tango", rust_lib_name("tango"))
        .library_search_path(cwd())
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg("--write-doc-meta-dir=parts-wrong")
        .run();

    htmldocck().arg(&out_dir).arg("tango.rs").run();
    htmldocck().arg(&out_dir).arg("quebec.rs").run();
    htmldocck().arg(&out_dir).arg("sierra.rs").run();
}

//@ needs-target-std

use run_make_support::{cwd, htmldocck, path, rust_lib_name, rustc, rustdoc};

fn main() {
    let merged_dir = path("merged");
    let parts_out_dir = path("parts");
    let lib_dir = path("lib");
    let out_dir = path("out");

    rustc().input("quebec.rs").crate_name("quebec").crate_type("rlib").run();
    rustdoc()
        .input("quebec.rs")
        .crate_name("quebec")
        .out_dir("quebec-out")
        .arg("-Zunstable-options")
        .arg(format!("--write-doc-meta-dir=info/doc.parts/quebec"))
        .arg("--enable-index-page")
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
        .out_dir("tango-out")
        .arg("-Zunstable-options")
        .arg(format!("--write-doc-meta-dir=info/doc.parts/tango"))
        .arg("--enable-index-page")
        .run();

    rustc()
        .input("sierra.rs")
        .crate_name("sierra")
        .crate_type("rlib")
        .extern_("tango", rust_lib_name("tango"))
        .library_search_path(cwd())
        .run();
    rustdoc()
        .input("sierra.rs")
        .crate_name("sierra")
        .extern_("tango", rust_lib_name("tango"))
        .library_search_path(cwd())
        .out_dir("sierra-out")
        .arg("-Zunstable-options")
        .arg(format!("--write-doc-meta-dir=info/doc.parts/sierra"))
        .arg("--enable-index-page")
        .run();

    rustc()
        .input("romeo.rs")
        .crate_name("romeo")
        .crate_type("rlib")
        .extern_("sierra", rust_lib_name("sierra"))
        .library_search_path(cwd())
        .run();
    rustdoc()
        .input("romeo.rs")
        .crate_name("romeo")
        .extern_("sierra", rust_lib_name("sierra"))
        .library_search_path(cwd())
        .out_dir("romeo-out")
        .arg("-Zunstable-options")
        .arg(format!("--write-doc-meta-dir=info/doc.parts/romeo"))
        .arg("--enable-index-page")
        .run();

    rustdoc()
        .input("indigo.rs")
        .crate_name("indigo")
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg(format!("--write-doc-meta-dir=info/doc.parts/indigo"))
        .arg("--enable-index-page")
        .run();

    let output = rustdoc()
        .arg("-Zunstable-options")
        .out_dir(&out_dir)
        .arg("--read-doc-meta-dir=info/doc.parts/tango")
        .arg("--read-doc-meta-dir=info/doc.parts/romeo")
        .arg("--read-doc-meta-dir=info/doc.parts/quebec")
        .arg("--read-doc-meta-dir=info/doc.parts/sierra")
        .arg("--read-doc-meta-dir=info/doc.parts/indigo")
        .arg("--enable-index-page")
        .run();
    output.assert_stderr_not_contains("error: the compiler unexpectedly panicked. this is a bug.");

    htmldocck().arg(&out_dir).arg("tango.rs").run();
    htmldocck().arg(&out_dir).arg("quebec.rs").run();
    htmldocck().arg(&out_dir).arg("sierra.rs").run();
    htmldocck().arg(&out_dir).arg("indigo.rs").run();
    htmldocck().arg(&out_dir).arg("romeo.rs").run();
}

//@ ignore-cross-compile

use run_make_support::{htmldocck, rust_lib_name, rustc, rustdoc};

fn main() {
    let out_dir = "rustdoc";
    let ex_dir = "ex.calls";
    let proc_crate_name = "foobar_macro";
    let crate_name = "foobar";

    let dylib_name = rustc()
        .crate_name(proc_crate_name)
        .crate_type("dylib")
        .print("file-names")
        .arg("-")
        .run()
        .stdout_utf8();

    rustc()
        .input("src/proc.rs")
        .crate_name(proc_crate_name)
        .edition("2021")
        .crate_type("proc-macro")
        .emit("dep-info,link")
        .run();
    rustc()
        .input("src/lib.rs")
        .crate_name(crate_name)
        .edition("2021")
        .crate_type("lib")
        .emit("dep-info,link")
        .run();

    rustdoc()
        .input("examples/ex.rs")
        .crate_name("ex")
        .crate_type("bin")
        .out_dir(&out_dir)
        .extern_(crate_name, rust_lib_name(crate_name))
        .extern_(proc_crate_name, dylib_name.trim())
        .arg("-Zunstable-options")
        .arg("--scrape-examples-output-path")
        .arg(&ex_dir)
        .arg("--scrape-examples-target-crate")
        .arg(crate_name)
        .run();

    rustdoc()
        .input("src/lib.rs")
        .crate_name(crate_name)
        .crate_type("lib")
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg("--with-examples")
        .arg(&ex_dir)
        .run();

    htmldocck().arg(out_dir).arg("src/lib.rs").run();
}

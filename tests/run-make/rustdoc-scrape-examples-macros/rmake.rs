//@ ignore-cross-compile

use run_make_support::{htmldocck, rustc, rustdoc, tmp_dir};

fn main() {
    let tmp_dir = tmp_dir();
    let out_dir = tmp_dir.join("rustdoc");
    let ex_dir = tmp_dir.join("ex.calls");
    let proc_crate_name = "foobar_macro";
    let crate_name = "foobar";

    let dylib_name = String::from_utf8(
        rustc()
            .crate_name(proc_crate_name)
            .crate_type("dylib")
            .arg("--print")
            .arg("file-names")
            .arg("-")
            .command_output()
            .stdout,
    )
    .unwrap();

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
        .output(&out_dir)
        .extern_(crate_name, tmp_dir.join(format!("lib{crate_name}.rlib")))
        .extern_(proc_crate_name, tmp_dir.join(dylib_name.trim()))
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
        .output(&out_dir)
        .arg("-Zunstable-options")
        .arg("--with-examples")
        .arg(&ex_dir)
        .run();

    assert!(htmldocck().arg(out_dir).arg("src/lib.rs").status().unwrap().success());
}

// Running --merge=finalize without an input crate root should not trigger ICE.
// Issue: https://github.com/rust-lang/rust/issues/146646

//@ needs-target-std

use run_make_support::{htmldocck, path, rustdoc};

fn main() {
    let out_dir = path("out");
    let merged_dir = path("merged");
    let parts_out_dir = path("parts");

    rustdoc()
        .input("dep1.rs")
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg(format!("--parts-out-dir={}", parts_out_dir.display()))
        .arg("--merge=none")
        .run();
    assert!(parts_out_dir.join("dep1.json").exists());

    rustdoc()
        .input("dep2.rs")
        .out_dir(&out_dir)
        .arg("-Zunstable-options")
        .arg(format!("--parts-out-dir={}", parts_out_dir.display()))
        .arg("--merge=none")
        .run();
    assert!(parts_out_dir.join("dep2.json").exists());

    // dep_missing is different, because --parts-out-dir is not supplied
    rustdoc().input("dep_missing.rs").out_dir(&out_dir).run();
    assert!(parts_out_dir.join("dep2.json").exists());

    let output = rustdoc()
        .arg("-Zunstable-options")
        .out_dir(&out_dir)
        .arg(format!("--include-parts-dir={}", parts_out_dir.display()))
        .arg("--merge=finalize")
        .run();
    output.assert_stderr_not_contains("error: the compiler unexpectedly panicked. this is a bug.");

    htmldocck().arg(&out_dir).arg("dep1.rs").run();
    htmldocck().arg(&out_dir).arg("dep2.rs").run();
    htmldocck().arg(out_dir).arg("dep_missing.rs").run();
}

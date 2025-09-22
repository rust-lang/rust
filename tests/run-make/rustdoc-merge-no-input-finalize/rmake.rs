// Running --merge=finalize without an input crate root should not trigger ICE.
// Issue: https://github.com/rust-lang/rust/issues/146646

//@ needs-target-std

use run_make_support::rustdoc;

fn main() {
    rustdoc()
        .input("sierra.rs")
        .arg("-Zunstable-options")
        .arg("--parts-out-dir=parts")
        .arg("--merge=none")
        .run();

    rustdoc()
        .arg("-Zunstable-options")
        .arg("--include-parts-dir=parts")
        .arg("--merge=finalize")
        .out_dir("out")
        .run();
}

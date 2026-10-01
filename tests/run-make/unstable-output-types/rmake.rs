// FIXME: remove the ignore gcc once fixed.
//@ ignore-backends: gcc

use std::ops::Not;
use std::path::Path;

use run_make_support::{invalid_utf8_contains, invalid_utf8_not_contains, rfs, rustc};

#[track_caller]
fn path_exists<const N: usize>(paths: [&str; N]) {
    for path in paths {
        assert!(Path::new(path).exists());
    }
}

#[track_caller]
fn path_not_exists<const N: usize>(paths: [&str; N]) {
    for path in paths {
        assert!(Path::new(path).exists().not());
    }
}

fn main() {
    rustc()
        .input("aux.rs")
        .crate_type("rlib")
        .emit("full-analysis")
        .arg("-Zunstable-options")
        .run();

    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .emit("dep-info,full-analysis")
        .arg("-Zbinary-dep-depinfo")
        .arg("-Zunstable-options")
        .run();

    invalid_utf8_contains("lib.d", "analysisaux.rmeta");
    invalid_utf8_not_contains("lib.d", "analysisaux.rlib");

    path_exists(["lib.d"]);

    rfs::remove_file("lib.d");
    rfs::remove_file("analysisaux.rmeta");

    rustc()
        .input("aux.rs")
        .crate_type("rlib")
        .emit("full-analysis")
        .arg("-Zunstable-options")
        .run();
    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .emit("full-analysis")
        .arg("-Zbinary-dep-depinfo")
        .arg("-Zunstable-options")
        .run();

    path_exists(["analysislib.rmeta"]);

    rfs::remove_file("analysislib.rmeta");

    // With nameres it doesn't lint, with analysis it does
    let output = rustc()
        .input("nameres.rs")
        .crate_type("rlib")
        .emit("dep-info,validation")
        .arg("-Dwarnings")
        .arg("-Zunstable-options")
        .run()
        .stderr_utf8();

    // Make sure that we don't run superfluous analysis:
    assert!(output.contains("dead_code").not());
    assert!(output.contains("named_asm_labels").not());

    // But the path still exists
    path_exists(["validationnameres.rmeta"]);

    rfs::remove_file("validationnameres.rmeta");

    let output = rustc()
        .input("nameres.rs")
        .crate_type("rlib")
        .emit("full-analysis")
        .arg("-Dwarnings")
        .arg("-Zunstable-options")
        .run_fail()
        .stderr_utf8();

    assert!(output.contains("dead_code"));
    assert!(output.contains("named_asm_labels"));

    // Won't exist because errors were emitted
    path_not_exists(["analysisnameres.rmeta"]);
}

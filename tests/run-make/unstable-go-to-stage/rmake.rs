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
    rustc().input("aux.rs").crate_type("rlib").emit("link,metadata").arg("-Zgo-to-stage=end").run();
    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .emit("dep-info")
        .arg("-Zbinary-dep-depinfo")
        .arg("-Zgo-to-stage=end")
        .run();

    invalid_utf8_contains("lib.d", "libaux.rmeta");
    invalid_utf8_not_contains("lib.d", "libaux.rlib");

    path_exists(["lib.d"]);

    rfs::remove_file("lib.d");
    rfs::remove_file("libaux.rmeta");

    rustc().input("aux.rs").crate_type("rlib").emit("dep-info").arg("-Zgo-to-stage=analysis").run();
    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .emit("dep-info")
        .arg("-Zbinary-dep-depinfo")
        .arg("-Zgo-to-stage=analysis") // These implies metadata
        .run();

    path_exists(["liblib.rmeta"]);

    rfs::remove_file("liblib.rmeta");

    // With nameres it doesn't lint, with analysis it does
    let output = rustc()
        .input("nameres.rs")
        .crate_type("rlib")
        .emit("dep-info,link")
        .arg("-Zgo-to-stage=nameres")
        .arg("-Dwarnings")
        .run()
        .stderr_utf8();

    // Make sure that we don't run superfluous analysis:
    assert!(output.contains("dead_code").not());
    assert!(output.contains("named_asm_labels").not());

    // But the path still exists
    path_exists(["libnameres.rmeta"]);

    let output = rustc()
        .input("nameres.rs")
        .crate_type("rlib")
        .emit("dep-info,link")
        .arg("-Zgo-to-stage=analysis")
        .arg("-Dwarnings")
        .run_fail()
        .stderr_utf8();

    // Make sure that we don't run superfluous analysis:
    assert!(output.contains("dead_code"));
    assert!(output.contains("named_asm_labels"));
}

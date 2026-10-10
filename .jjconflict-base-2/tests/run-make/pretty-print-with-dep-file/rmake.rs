//@ needs-target-std
//
// Passing --emit=dep-info to the Rust compiler should create a .d file...
// but it failed to do so in Rust 1.69.0 when combined with -Z unpretty=expanded
// due to a bug. This test checks that -Z unpretty=expanded does not prevent the
// generation of the dep-info file, and that its -Z unpretty=normal counterpart
// does not get an unexpected dep-info file.
// See https://github.com/rust-lang/rust/issues/112898

use run_make_support::{invalid_utf8_contains, path, rfs, rustc};

fn main() {
    rustc().emit("dep-info").arg("-Zunpretty=expanded").input("with-dep.rs").run();
    invalid_utf8_contains("with-dep.d", "with-dep.rs");
    rfs::remove_file("with-dep.d");
    rustc().emit("dep-info").arg("-Zunpretty=normal").input("with-dep.rs").run();
    assert!(!path("with-dep.d").exists());
}

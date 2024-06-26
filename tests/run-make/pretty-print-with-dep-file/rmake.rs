// Passing --emit=dep-info to the Rust compiler should create a .d file...
// but it failed to do so in Rust 1.69.0 when combined with -Z unpretty=expanded
// due to a bug. This test checks that -Z unpretty=expanded does not prevent the
// generation of the dep-info file, and that its -Z unpretty=normal counterpart
// does not get an unexpected dep-info file.
// See https://github.com/rust-lang/rust/issues/112898

use run_make_support::{fs_wrapper, invalid_utf8_contains, rustc};
use std::path::Path;

fn main() {
    rustc().emit("dep-info").arg("-Zunpretty=expanded").input("with-dep.rs").run();
    invalid_utf8_contains("with-dep.d", "with-dep.rs");
    fs_wrapper::remove_file("with-dep.d");
    rustc().emit("dep-info").arg("-Zunpretty=normal").input("with-dep.rs").run();
    assert!(!Path::new("with-dep.d").exists());
}

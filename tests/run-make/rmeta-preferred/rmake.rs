// This test compiles `lib.rs`'s dependency, `rmeta_aux.rs`, as both an rlib
// and an rmeta crate. By default, rustc should give the metadata crate (rmeta)
// precedence over the rust-lib (rlib). This test inspects the contents of the binary
// and that the correct (rmeta) crate was used.
// rlibs being preferred could indicate a resurgence of the -Zbinary-dep-depinfo bug
// seen in #68298.
// See https://github.com/rust-lang/rust/pull/37681

//@ ignore-cross-compile

use run_make_support::{invalid_utf8_contains, invalid_utf8_not_contains, rustc};

fn main() {
    rustc().input("rmeta_aux.rs").crate_type("rlib").emit("link,metadata").run();
    rustc().input("lib.rs").crate_type("rlib").emit("dep-info").arg("-Zbinary-dep-depinfo").run();
    invalid_utf8_contains("lib.d", "librmeta_aux.rmeta");
    invalid_utf8_not_contains("lib.d", "librmeta_aux.rlib");
}

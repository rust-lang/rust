//@ ignore-cross-compile

// NOTE: `sdylib`'s platform support is basically just `dylib`'s platform support.
//@ needs-crate-type: dylib

use run_make_support::rustc;

fn main() {
    rustc().env("RUSTC_FORCE_RUSTC_VERSION", "1").input("libr.rs").run();
    rustc().env("RUSTC_FORCE_RUSTC_VERSION", "2").input("app.rs").run();
}

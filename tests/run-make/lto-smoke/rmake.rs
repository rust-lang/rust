// A simple smoke test to check that link time optimization
// (LTO) works as intended, with its various flags turned on.
// See https://github.com/rust-lang/rust/issues/10741

//@ ignore-cross-compile

use run_make_support::rustc;

fn main() {
    let lto_flags = ["-Clto", "-Clto=yes", "-Clto=off", "-Clto=thin", "-Clto=fat"];
    for flag in lto_flags {
        rustc().input(lib.rs).run();
        rustc().input(main.rs).arg(flag).run();
    }
}

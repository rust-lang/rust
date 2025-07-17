// This test ensures that the “symbol not found” error does not occur
// when the symbols in the `PROVIDE` of the link script can be eliminated.
// This is a regression test for #131164.

//@ needs-rust-lld
//@ only-x86_64-unknown-linux-gnu

use run_make_support::rustc;

fn main() {
    rustc()
        .input("main.rs")
        .arg("-Clinker-features=+lld")
        .arg("-Clink-self-contained=+linker")
        .arg("-Zunstable-options")
        .link_arg("-Tscript.t")
        .run();
}

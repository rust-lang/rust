//@ only-x86_64-pc-windows-msvc
//@ needs-rust-lld

use run_make_support::rustc;

fn main() {
    // Ensure that LLD can link when an .rlib contains a synthetic object
    // file referencing exported or used symbols.
    rustc().input("foo.rs").linker("rust-lld").run();

    // Ensure that LLD can link when /WHOLEARCHIVE: is used with an .rlib.
    // Previously, lib.rmeta was not marked as (trivially) SAFESEH-aware.
    rustc().input("baz.rs").run();
    rustc().input("bar.rs").linker("rust-lld").link_arg("/WHOLEARCHIVE:libbaz.rlib").run();
}

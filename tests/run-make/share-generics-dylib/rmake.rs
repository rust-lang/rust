//@ needs-target-std
//
// This test makes sure all generic instances get re-exported from Rust dylibs for use by
// `-Zshare-generics`. There are two rlibs (`instance_provider_a` and `instance_provider_b`)
// which both provide an instance of `Cell<i32>::set`. There is `instance_user_dylib` which is
// supposed to re-export both these instances, and then there are `instance_user_a_rlib` and
// `instance_user_b_rlib` which each rely on a specific instance to be available.
//
// In the end everything is linked together into `linked_leaf`. If `instance_user_dylib` does
// not export both then we'll get an `undefined reference` error for one of the instances.
//
// This is regression test for https://github.com/rust-lang/rust/issues/67276.

use run_make_support::rustc;

fn main() {
    compile("rlib", "instance_provider_a.rs");
    compile("rlib", "instance_provider_b.rs");
    compile("dylib", "instance_user_dylib.rs");
    compile("rlib", "instance_user_a_rlib.rs");
    compile("rlib", "instance_user_b_rlib.rs");
    compile("bin", "linked_leaf.rs");
}

fn compile(crate_type: &str, input: &str) {
    rustc()
        .input(input)
        .crate_type(crate_type)
        .args(&["-Cprefer-dynamic", "-Zshare-generics=yes", "-Csymbol-mangling-version=v0"])
        .codegen_units(1)
        .run();
}

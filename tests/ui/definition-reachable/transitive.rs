//@ aux-build:transitive-macro.rs
//@ build-fail

extern crate transitive_macro;

fn main() {
    transitive_macro::m!();
}

//~? ERROR missing optimized MIR for `transitive_macro::mod1::mod2::foo` in the crate `transitive_macro`

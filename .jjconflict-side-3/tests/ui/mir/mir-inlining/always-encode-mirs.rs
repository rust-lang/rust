// Regression test for MIR inlining with -Zalways-encode-mir enabled in the auxiliary crate.
// Previously we inlined function not eligible for inlining which lead to linking error:
// undefined reference to `internal::S`
//
//@ aux-build:internal.rs
//@ build-pass
//@ compile-flags: -O
extern crate internal;

fn main() {
    println!("{}", internal::f());
}

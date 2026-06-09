//@ aux-build:unstable_impl_codegen_aux2.rs
//@ run-pass

/// Downstream crate for unstable impl codegen test
/// that depends on upstream crate in
/// unstable_impl_codegen_aux2.rs

extern crate unstable_impl_codegen_aux2 as aux;
use aux::foo;

fn main() {
    foo(1_u8);
}

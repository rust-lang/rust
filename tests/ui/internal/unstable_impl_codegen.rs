//@ aux-build:unstable_impl_codegen_aux2.rs

/// Downstream crate for unstable impl codegen test 
/// that depends on upstream crate in 
/// unstable_impl_codegen_aux2.rs

fn main() {
    foo(1_u8);
}

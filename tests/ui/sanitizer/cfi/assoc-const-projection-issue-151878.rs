//@ compile-flags: -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer -Ccodegen-units=1 -Clto
//@ needs-rustc-debug-assertions
//@ needs-sanitizer-cfi
//@ build-pass
//@ no-prefer-dynamic

#![feature(gca_min_const_items, gca_macroless_args, associated_type_defaults)]
#![expect(incomplete_features)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const N: usize = gca!(0);
    fn process(&self, _: [u8; Self::N]) {}
}

impl Trait for () {}

fn main() {
    let _x: &dyn Trait<N = 0> = &();
}

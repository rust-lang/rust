//@ compile-flags: -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer -Ccodegen-units=1 -Clto
//@ needs-rustc-debug-assertions
//@ needs-sanitizer-cfi
//@ build-pass
//@ no-prefer-dynamic

#![feature(min_generic_const_args, associated_type_defaults)]
#![expect(incomplete_features)]

trait Trait {
    type const N: usize = 0;
    fn process(&self, _: [u8; Self::N]) {}
}

impl Trait for () {}

fn main() {
    let _x: &dyn Trait<N = 0> = &();
}

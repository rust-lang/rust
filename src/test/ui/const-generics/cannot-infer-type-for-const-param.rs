#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

// We should probably be able to infer the types here. However, this test is checking that we don't
// get an ICE in this case. It may be modified later to not be an error.

struct Foo<const NUM_BYTES: usize>(pub [u8; NUM_BYTES]);

fn main() {
    let _ = Foo::<3>([1, 2, 3]); //~ ERROR type annotations needed
}

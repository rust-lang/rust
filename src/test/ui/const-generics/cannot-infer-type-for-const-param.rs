// build-pass (FIXME(62277): could be check-pass?)
#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

// This test confirms that the types can be inferred correctly for this example with const
// generics. Previously this would ICE, and more recently error.

struct Foo<const NUM_BYTES: usize>(pub [u8; NUM_BYTES]);

fn main() {
    let _ = Foo::<3>([1, 2, 3]); //~ ERROR type annotations needed
    //~^ ERROR mismatched types
}

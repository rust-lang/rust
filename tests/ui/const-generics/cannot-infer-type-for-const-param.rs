//@ check-pass

// This test confirms that the types can be inferred correctly for this example with const
// generics. Previously this would ICE, and more recently error.

struct Foo<const NUM_BYTES: usize>(pub [u8; NUM_BYTES]);

fn main() {
    let _ = Foo::<3>([1, 2, 3]);
}

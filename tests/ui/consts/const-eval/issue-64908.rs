//@ run-pass

// This test verifies that the `ConstProp` pass doesn't cause an ICE when evaluating polymorphic
// promoted MIR.

pub trait ArrowPrimitiveType {
    type Native;
}

pub fn new<T: ArrowPrimitiveType>() {
    assert_eq!(0, std::mem::size_of::<T::Native>());
}

impl ArrowPrimitiveType for () {
    type Native = ();
}

fn main() {
    new::<()>();
}

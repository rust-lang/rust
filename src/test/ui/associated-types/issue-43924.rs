#![feature(associated_type_defaults)]

// This used to cause an ICE because assoc. type defaults weren't properly
// type-checked.

trait Foo<T: Default + ToString> {
    type Out: Default + ToString + ?Sized = dyn ToString;  //~ error: not satisfied
}

impl Foo<u32> for () {}  //~ error: not satisfied
impl Foo<u64> for () {}  //~ error: not satisfied

fn main() {
    assert_eq!(<() as Foo<u32>>::Out::default().to_string(), "false");
}

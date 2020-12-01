#![feature(associated_type_defaults)]

// This used to cause an ICE because assoc. type defaults weren't properly
// type-checked.

trait Foo<T: Default + ToString> {
    type Out: Default + ToString + ?Sized = dyn ToString;  //~ ERROR not satisfied
}

impl Foo<u32> for () {}
impl Foo<u64> for () {}

fn main() {
    assert_eq!(<() as Foo<u32>>::Out::default().to_string(), "false");
    //~^ ERROR no function or associated item named `default` found for trait object
}

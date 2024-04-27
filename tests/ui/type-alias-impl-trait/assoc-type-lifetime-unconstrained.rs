// Tests that we don't allow unconstrained lifetime parameters in impls when
// the lifetime is used in an associated opaque type.

#![feature(impl_trait_in_assoc_type)]

trait UnwrapItemsExt {
    type Iter;
    fn unwrap_items(self) -> Self::Iter;
}

struct MyStruct {}

trait MyTrait<'a> {}

impl<'a> MyTrait<'a> for MyStruct {}

impl<'a, I> UnwrapItemsExt for I {
    //~^ ERROR the lifetime parameter `'a` is not constrained
    type Iter = impl MyTrait<'a>;

    fn unwrap_items(self) -> Self::Iter {
        MyStruct {}
        //~^ ERROR expected generic lifetime parameter
    }
}

fn main() {}

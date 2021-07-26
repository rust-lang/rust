// Tests that we still detect defining usages when
// lifetimes are used in an associated opaque type
// check-pass

#![feature(type_alias_impl_trait)]

trait UnwrapItemsExt<'a> {
    type Iter;
    fn unwrap_items(self) -> Self::Iter;
}

struct MyStruct {}

trait MyTrait<'a> {}

impl<'a> MyTrait<'a> for MyStruct {}

impl<'a, I> UnwrapItemsExt<'a> for I {
    type Iter = impl MyTrait<'a>;

    fn unwrap_items(self) -> Self::Iter {
        MyStruct {}
    }
}

fn main() {}

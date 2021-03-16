// Tests that we still detect defining usages when
// lifetimes are used in an associated opaque type
// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

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

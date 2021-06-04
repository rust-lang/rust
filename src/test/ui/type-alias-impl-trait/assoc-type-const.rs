// Tests that we properly detect defining usages when using
// const generics in an associated opaque type
// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

trait UnwrapItemsExt<'a, const C: usize> {
    type Iter;
    fn unwrap_items(self) -> Self::Iter;
}

struct MyStruct<const C: usize> {}

trait MyTrait<'a, const C: usize> {
    type MyItem;
    const MY_CONST: usize;
}

impl<'a, const C: usize> MyTrait<'a, C> for MyStruct<C> {
    type MyItem = u8;
    const MY_CONST: usize = C;
}

impl<'a, I, const C: usize> UnwrapItemsExt<'a, C> for I {
    type Iter = impl MyTrait<'a, C>;

    fn unwrap_items(self) -> Self::Iter {
        MyStruct::<C> {}
    }
}

fn main() {}

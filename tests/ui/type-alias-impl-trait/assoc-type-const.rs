// Tests that we properly detect defining usages when using
// const generics in an associated opaque type

//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(impl_trait_in_assoc_type)]

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

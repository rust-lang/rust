//@ compile-flags: --crate-type=lib
//@ revisions: current next current_sized_hierarchy next_sized_hierarchy
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass
//@[next] check-pass
//@[next] compile-flags: -Znext-solver
//@[next_sized_hierarchy] compile-flags: -Znext-solver

#![cfg_attr(any(current_sized_hierarchy, next_sized_hierarchy), feature(sized_hierarchy))]

// Test that we avoid incomplete inference when normalizing. Without this,
// `Trait`'s implicit `MetaSized` supertrait requires proving `T::Assoc<_>: MetaSized`
// before checking the `new` arguments, resulting in eagerly constraining the inference
// var to `u32`. This is undesirable and would breaking code.

pub trait Trait {
    type Assoc<G>: OtherTrait<G>;
}

pub trait OtherTrait<R> {
    fn new(r: R) -> R {
        r
    }
}

pub fn function<T: Trait>()
where
    T::Assoc<[u32; 1]>: Clone,
{
    let _x = T::Assoc::new(());
    //[next_sized_hierarchy]~^ ERROR mismatched types
    //[current_sized_hierarchy]~^^ ERROR mismatched types
}

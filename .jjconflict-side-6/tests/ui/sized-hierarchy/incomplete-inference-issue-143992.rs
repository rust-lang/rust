//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

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
}

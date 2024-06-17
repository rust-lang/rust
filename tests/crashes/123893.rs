//@ known-bug: #123893
//@ compile-flags: -Zpolymorphize=on -Zinline-mir=yes -Zinline-mir-threshold=20
pub fn main() {
    generic_impl::<bool>();
}

fn generic_impl<T>() {
    trait MagicTrait {
        const IS_BIG: bool;
    }
    impl<T> MagicTrait for T {
        const IS_BIG: bool = true;
    }
    more_cost();
    if T::IS_BIG {
        big_impl::<i32>();
    }
}

#[inline(never)]
fn big_impl<T>() {}

#[inline(never)]
fn more_cost() {}

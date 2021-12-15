#![feature(const_fn_trait_bound, generic_const_exprs)]
#![allow(incomplete_features)]

trait MiniTypeId {
    const TYPE_ID: u64;
}

impl<T> MiniTypeId for T {
    const TYPE_ID: u64 = 0;
}

enum Lift<const V: bool> {}

trait IsFalse {}
impl IsFalse for Lift<false> {}

const fn is_same_type<T: MiniTypeId, U: MiniTypeId>() -> bool {
    T::TYPE_ID == U::TYPE_ID
}

fn requires_distinct<A, B>(_a: A, _b: B) where
    A: MiniTypeId, B: MiniTypeId,
    Lift<{is_same_type::<A, B>()}>: IsFalse {}

fn main() {
    requires_distinct("str", 12);
    //~^ ERROR mismatched types
}

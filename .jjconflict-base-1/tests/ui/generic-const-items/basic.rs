//@ check-pass

// Basic usage patterns of free & associated generic const items.

#![feature(generic_const_items)]
#![allow(incomplete_features)]

fn main() {
    const NULL<T>: Option<T> = None::<T>;
    const NOTHING<T>: Option<T> = None; // arg inferred

    let _ = NOTHING::<String>;
    let _: Option<u8> = NULL; // arg inferred

    const IDENTITY<const X: u64>: u64 = X;

    const COUNT: u64 = IDENTITY::<48>;
    const AMOUNT: u64 = IDENTITY::<COUNT>;
    const NUMBER: u64 = IDENTITY::<{ AMOUNT * 2 }>;
    let _ = NUMBER;
    let _ = IDENTITY::<0>;

    let _ = match 0 {
        IDENTITY::<1> => 2,
        IDENTITY::<{ 1 + 1 }> => 4,
        _ => 0,
    };

    const CREATE<I: Inhabited>: I = I::PROOF;
    let _ = CREATE::<u64>;
    let _: u64 = CREATE; // arg inferred

    let _ = <() as Main<u64>>::MAKE::<u64>;
    let _: (u64, u64) = <()>::MAKE; // args inferred
}

pub fn usage<'any>() {
    const REGION_POLY<'a>: &'a () = &();

    let _: &'any () = REGION_POLY::<'any>;
    let _: &'any () = REGION_POLY::<'_>;
    let _: &'static () = REGION_POLY;
}

trait Main<O> {
    type Output<I>;
    const MAKE<I: Inhabited>: Self::Output<I>;
}

impl<O: Inhabited> Main<O> for () {
    type Output<I> = (O, I);
    const MAKE<I: Inhabited>: Self::Output<I> = (O::PROOF, I::PROOF);
}

trait Inhabited {
    const PROOF: Self;
}

impl Inhabited for u64 {
    const PROOF: Self = 512;
}

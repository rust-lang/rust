//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(min_specialization, const_trait_impl)]

struct Dummy;

const trait DummyTrait {
    fn dummy_fn() -> u32;
}
impl DummyTrait for Wrap<i32> {
    fn dummy_fn() -> u32 {
        println!("wut");
        0
    }
}

const trait Trait {
    fn trait_fn() -> u32;
}
impl<T> const Trait for T where T: DummyTrait {
    default fn trait_fn() -> u32 {
        42
    }
}

struct Wrap<T>(T);

impl<T> const Trait for Wrap<T> where Self: [const] DummyTrait {
    fn trait_fn() -> u32 {
        <Wrap<T>>::dummy_fn()
    }
}

const fn indirect<T: DummyTrait>() -> u32 {
    T::trait_fn()
}

const A: u32 = indirect::<Wrap<i32>>();

const B: () = { assert!(A == 42); };

fn main() {}

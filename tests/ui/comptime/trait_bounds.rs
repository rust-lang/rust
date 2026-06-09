#![feature(rustc_attrs, const_trait_impl)]

const trait Trait {
    fn method() {}
}

#[rustc_comptime]
const fn always_const<T: const Trait>() {
    T::method()
}

#[rustc_comptime]
const fn conditionally_const<T: [const] Trait>() {
    T::method()
    //~^ ERROR: `T: const Trait` is not satisfied
}

#[rustc_comptime]
const fn non_const<T: Trait>() {
    T::method()
    //~^ ERROR: `T: const Trait` is not satisfied
}

fn main() {}

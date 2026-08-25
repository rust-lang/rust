#![feature(rustc_attrs, const_trait_impl)]

const trait Trait {
    fn method() {}
}

#[rustc_comptime]
fn always_const<T: const Trait>() {
    T::method()
}

#[rustc_comptime]
fn conditionally_const<T: [const] Trait>() {
    //~^ ERROR: `[const]` is not allowed here
    T::method()
    //~^ ERROR: `T: const Trait` is not satisfied
}

#[rustc_comptime]
fn non_const<T: Trait>() {
    T::method()
    //~^ ERROR: `T: const Trait` is not satisfied
}

fn main() {}

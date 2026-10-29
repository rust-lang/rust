#![feature(comptime, const_trait_impl)]

const trait Trait {
    fn method() {}
}

#[comptime]
fn always_const<T: const Trait>() {
    T::method()
}

#[comptime]
fn conditionally_const<T: [const] Trait>() {
    //~^ ERROR: `[const]` is not allowed here
    T::method()
    //~^ ERROR: `T: const Trait` is not satisfied
}

#[comptime]
fn non_const<T: Trait>() {
    T::method()
    //~^ ERROR: `T: const Trait` is not satisfied
}

fn main() {}

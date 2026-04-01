// Check that we disallow GAT param defaults, even with `invalid_type_param_default` allowed

#![allow(invalid_type_param_default)]

trait Trait {
    type Assoc<T = u32>;
    //~^ ERROR defaults for generic parameters are not allowed here
}

impl Trait for () {
    type Assoc<T = u32> = u64;
    //~^ ERROR defaults for generic parameters are not allowed here
}

impl Trait for u32 {
    type Assoc<T = u32> = T;
    //~^ ERROR defaults for generic parameters are not allowed here
}

trait Other {}
impl Other for u32 {}

fn foo<T>()
where
    T: Trait<Assoc = u32>,
    T::Assoc: Other {
    }

fn main() {
    // errors
    foo::<()>();
    //~^ ERROR type mismatch
    //~| ERROR `u64: Other` is not satisfied
    // works
    foo::<u32>();
}

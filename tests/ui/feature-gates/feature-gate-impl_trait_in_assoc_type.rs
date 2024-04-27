trait Foo {
    type Bar;
}

impl Foo for () {
    type Bar = impl std::fmt::Debug;
    //~^ ERROR: `impl Trait` in associated types is unstable
    //~| ERROR: unconstrained opaque type
}

struct Mop;

impl Mop {
    type Bop = impl std::fmt::Debug;
    //~^ ERROR: `impl Trait` in associated types is unstable
    //~| ERROR: inherent associated types are unstable
    //~| ERROR: unconstrained opaque type
}

fn main() {}

use std::fmt::Debug;

type Foo = impl Debug; //~ ERROR `impl Trait` in type aliases is unstable

trait Bar {
    type Baa: Debug;
    fn define() -> Self::Baa;
}

impl Bar for () {
    type Baa = impl Debug; //~ ERROR `impl Trait` in type aliases is unstable
    fn define() -> Self::Baa { 0 }
}

fn define() -> Foo { 0 }

trait TraitWithDefault {
    type Assoc = impl Debug;
    //~^ ERROR associated type defaults are unstable
    //~| ERROR `impl Trait` not allowed outside of function
    //~| ERROR `impl Trait` in type aliases is unstable
}

type NestedFree = (Vec<impl Debug>, impl Debug, impl Iterator<Item = impl Debug>);
//~^ ERROR `impl Trait` in type aliases is unstable
//~| ERROR `impl Trait` in type aliases is unstable
//~| ERROR `impl Trait` in type aliases is unstable
//~| ERROR `impl Trait` in type aliases is unstable
//~| ERROR `impl Trait` not allowed outside of function
//~| ERROR `impl Trait` not allowed outside of function
//~| ERROR `impl Trait` not allowed outside of function

impl Bar for u8 {
    type Baa = (Vec<impl Debug>, impl Debug, impl Iterator<Item = impl Debug>);
    //~^ ERROR `impl Trait` in type aliases is unstable
    //~| ERROR `impl Trait` in type aliases is unstable
    //~| ERROR `impl Trait` in type aliases is unstable
    //~| ERROR `impl Trait` in type aliases is unstable
    //~| ERROR `impl Trait` not allowed outside of function
    //~| ERROR `impl Trait` not allowed outside of function
    //~| ERROR `impl Trait` not allowed outside of function
    fn define() -> Self::Baa { (vec![true], 0u8, 0i32..1) }
}

fn main() {}

#![feature(const_trait_impl, trait_alias, comptime)]

#[comptime]
//~^ ERROR: only functions, trait impls, and methods may be comptime
trait Trait {
    fn method(&self) {}
}

const impl Trait for () {}
//~^ ERROR: const `impl` for trait `Trait` which is not `const`

#[comptime]
impl Trait for u32 {
    //~^ ERROR: comptime `impl` for trait `Trait` which is not `const`
    fn method(&self) {
        comptime_fn();
    }
}

#[comptime]
fn comptime_fn() {}

#[comptime]
//~^ ERROR: only functions, trait impls, and methods may be comptime
trait TraitAlias = const Trait;
//~^ ERROR: `const` can only be applied to `const` traits
//~| ERROR: `const` can only be applied to `const` traits
//~| ERROR: `const` can only be applied to `const` traits

#[comptime]
fn func<T: const TraitAlias>(t: &T) {
    //~^ ERROR: `const` can only be applied to `const` traits
    t.method()
    //~^ ERROR: cannot call non-const method `<T as Trait>::method` in constants
}

fn main() {}

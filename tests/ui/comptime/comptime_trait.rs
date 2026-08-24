#![feature(rustc_attrs, const_trait_impl, trait_alias)]

#[rustc_comptime]
//~^ ERROR: the `rustc_comptime` attribute cannot be used on traits
trait Trait {
    fn method(&self) {}
}

const impl Trait for () {}

#[rustc_comptime]
//~^ ERROR: the `rustc_comptime` attribute cannot be used on trait impl
impl Trait for u32 {
    fn method(&self) {
        comptime_fn();
    }
}

#[rustc_comptime]
fn comptime_fn() {}

#[rustc_comptime]
//~^ ERROR: the `rustc_comptime` attribute cannot be used on trait aliases
trait TraitAlias = const Trait;

#[rustc_comptime]
fn func<T: const TraitAlias>(t: &T) {
    t.method()
    //~^ ERROR: cannot call non-const method `<T as Trait>::method` in constants
}

fn main() {}

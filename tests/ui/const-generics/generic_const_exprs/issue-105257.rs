#![feature(generic_const_exprs)]
#![expect(incomplete_features)]

trait Trait<T> {
    fn fnc<const N: usize = "">(&self) {} //~ERROR defaults for generic parameters are not allowed here
    //~^ ERROR: mismatched types
    fn foo<const N: usize = { std::mem::size_of::<T>() }>(&self) {} //~ERROR defaults for generic parameters are not allowed here
}

fn main() {}

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

fn function() -> u32 {
    17
}

struct Wrapper<const F: fn() -> u32>; //~ ERROR: using function pointers as const generic parameters

impl<const F: fn() -> u32> Wrapper<F> {
//~^ ERROR: using function pointers as const generic parameters
    fn call() -> u32 {
        F()
    }
}

fn main() {
    assert_eq!(Wrapper::<function>::call(), 17);
}

// Check that functions cannot be used as const parameters.
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

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

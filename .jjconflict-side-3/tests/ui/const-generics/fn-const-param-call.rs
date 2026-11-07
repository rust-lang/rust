// Check that functions cannot be used as const parameters.
//@ revisions: min adt_const_params full

#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(adt_const_params, feature(adt_const_params))]
#![cfg_attr(adt_const_params, allow(incomplete_features))]

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

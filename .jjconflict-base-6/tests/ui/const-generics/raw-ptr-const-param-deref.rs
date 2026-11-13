// Checks that pointers must not be used as the type of const params.
//@ revisions: min adt_const_params full

#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(adt_const_params, feature(adt_const_params))]
#![cfg_attr(adt_const_params, allow(incomplete_features))]

const A: u32 = 3;

struct Const<const P: *const u32>; //~ ERROR: using raw pointers as const generic parameters

impl<const P: *const u32> Const<P> {
    //~^ ERROR: using raw pointers as const generic parameters
    fn get() -> u32 {
        unsafe { *P }
    }
}

fn main() {
    assert_eq!(Const::<{ &A as *const _ }>::get(), 3)
}

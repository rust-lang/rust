// revisions: simple adt_const_params
#![cfg_attr(adt_const_params, feature(adt_const_params))]
#![cfg_attr(adt_const_params, allow(incomplete_features))]

fn foo<const F: f32>() {}
//~^ ERROR `f32` is forbidden as the type of a const generic parameter

const C: f32 = 1.0;

fn main() {
    foo::<C>();
}

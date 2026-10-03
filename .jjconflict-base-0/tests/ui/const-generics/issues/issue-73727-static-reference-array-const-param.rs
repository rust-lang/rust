// Regression test for #73727

//@ revisions: full min
//@[full]check-pass

#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

fn a<const X: &'static [u32]>() {}
//[min]~^ ERROR `&'static [u32]` is forbidden as the type of a const generic parameter

fn main() {
    a::<{ &[] }>();
}

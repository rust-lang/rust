//@ check-pass
// (this requires debug assertions)

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

fn foo<const B: &'static bool>(arg: &'static bool) -> bool {
    B == arg
}

fn main() {
    foo::<{ &true }>(&false);
}

// check-pass
// (this requires debug assertions)

#![feature(adt_const_params)]
#![allow(incomplete_features, ref_binop_on_copy_type)]

fn foo<const B: &'static bool>(arg: &'static bool) -> bool {
    B == arg
}

fn main() {
    foo::<{ &true }>(&false);
}

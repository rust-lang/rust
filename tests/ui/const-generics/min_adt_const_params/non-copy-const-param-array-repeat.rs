//@ run-pass

#![feature(const_param_ty_trait, min_adt_const_params)]

use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
struct NonCopy(u32);

impl ConstParamTy_ for NonCopy {}

fn test<const A: NonCopy>() {
    let arr = [A; 17];
    for elem in arr {
        assert_eq!(elem.0, 42);
    }
}

fn main() {
    test::<{ NonCopy(42) }>();
}

// Tests that array sizes that depend on const-params are checked using `ConstEvaluatable`.
//@ revisions: full min

#![cfg_attr(full, feature(generic_const_exprs, adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

#[allow(dead_code)]
struct ArithArrayLen<const N: usize>([u32; 0 + N]);
//[full]~^ ERROR unconstrained generic constant
//[min]~^^ ERROR generic parameters may not be used in const operations

#[cfg(full)]
use std::marker::ConstParamTy;

#[derive(PartialEq, Eq)]
#[cfg_attr(full, derive(ConstParamTy))]
struct Config {
    arr_size: usize,
}

struct B<const CFG: Config> {
    //[min]~^ ERROR `Config` is forbidden
    arr: [u8; CFG.arr_size],
    //[full]~^ ERROR overly complex generic constant
    //[min]~^^ ERROR generic parameters may not be used in const operations
}

const C: Config = Config { arr_size: 5 };

fn main() {
    let b = B::<C> { arr: [1, 2, 3, 4, 5] };
    assert_eq!(b.arr.len(), 5);
}

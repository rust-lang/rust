// revisions: stock gated
// gate-test-const_trait_bound_opt_out

#![cfg_attr(gated, feature(const_trait_bound_opt_out))]
#![allow(incomplete_features)]

trait T {
    const CONST: i32;
}

const fn get_assoc_const<S: ?const T>() -> i32 { <S as T>::CONST }
//[stock]~^ ERROR `?const` on trait bounds is experimental
//[stock,gated]~^^ ERROR `?const` on trait bounds is not yet implemented

fn main() {}

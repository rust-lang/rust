#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

use std::gca;

struct S;

// FIXME(mgca): need support for ctors without anon const
// (we use a const-block to trigger an anon const here)
const FREE: S = gca!(const { S });
//~^ ERROR `S` must implement `ConstParamTy` to be used as the type of a const generic parameter

trait Tr {
    #[rustc_always_gca]
    const N: S;
    //~^ ERROR `S` must implement `ConstParamTy` to be used as the type of a const generic parameter
}

impl Tr for S {
    // FIXME(mgca): need support for ctors without anon const
    // (we use a const-block to trigger an anon const here)
    const N: S = gca!(const { S });
    //~^ ERROR `S` must implement `ConstParamTy` to be used as the type of a const generic parameter
}

fn main() {}

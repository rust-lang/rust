#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

struct S;

// FIXME(mgca): need support for ctors without anon const
// (we use a const-block to trigger an anon const here)
type const FREE: S = const { S };
//~^ ERROR `S` must implement `ConstParamTy` to be used as the type of a const generic parameter

trait Tr {
    type const N: S;
    //~^ ERROR `S` must implement `ConstParamTy` to be used as the type of a const generic parameter
}

impl Tr for S {
    // FIXME(mgca): need support for ctors without anon const
    // (we use a const-block to trigger an anon const here)
    type const N: S = const { S };
    //~^ ERROR `S` must implement `ConstParamTy` to be used as the type of a const generic parameter
}

fn main() {}

#![feature(adt_const_params, unsized_const_params, generic_const_parameter_types)]
#![expect(incomplete_features)]

use std::marker::{ConstParamTy_, PhantomData};

struct UsesType<T, const N: usize, const M: [T; N]>(PhantomData<T>);
//~^ ERROR: `[T; N]` can't be used as a const parameter type

fn main() {}

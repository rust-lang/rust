#![feature(adt_const_params, generic_const_parameter_types)]
#![expect(incomplete_features)]

use std::marker::PhantomData;

struct UsesConst<const N: [u8; M], const M: usize>;
//~^ ERROR: the type of const parameters must not depend on other generic parameters
struct UsesType<const N: [T; 2], T>(PhantomData<T>);
//~^ ERROR: the type of const parameters must not depend on other generic parameters

fn main() {}

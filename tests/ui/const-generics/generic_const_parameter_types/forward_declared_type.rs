#![feature(adt_const_params, generic_const_parameter_types)]
#![expect(incomplete_features)]

use std::marker::PhantomData;

struct UsesConst<const N: [u8; M], const M: usize>;
//~^ ERROR: const parameter types cannot reference parameters before they are declared
struct UsesType<const N: [T; 2], T>(PhantomData<T>);
//~^ ERROR: const parameter types cannot reference parameters before they are declared

fn main() {}

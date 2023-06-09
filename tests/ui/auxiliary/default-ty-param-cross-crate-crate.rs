#![crate_type = "lib"]
#![crate_name = "default_param_test"]
#![feature(default_type_parameter_fallback)]

use std::marker::PhantomData;

pub struct Foo<A, B>(PhantomData<(A, B)>);

pub fn bleh<A=i32, X=char>() -> Foo<A, X> { Foo(PhantomData) }

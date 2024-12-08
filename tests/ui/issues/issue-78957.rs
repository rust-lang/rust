#![deny(unused_attributes)]

use std::marker::PhantomData;

pub struct Foo<#[inline] const N: usize>;
//~^ ERROR attribute should be applied to function or closure
pub struct Bar<#[cold] const N: usize>;
//~^ ERROR attribute should be applied to a function
//~| WARN this was previously accepted
pub struct Baz<#[repr(C)] const N: usize>;
//~^ ERROR attribute should be applied to a struct, enum, or union
//
pub struct Foo2<#[inline] 'a>(PhantomData<&'a ()>);
//~^ ERROR attribute should be applied to function or closure
pub struct Bar2<#[cold] 'a>(PhantomData<&'a ()>);
//~^ ERROR attribute should be applied to a function
//~| WARN this was previously accepted
pub struct Baz2<#[repr(C)] 'a>(PhantomData<&'a ()>);
//~^ ERROR attribute should be applied to a struct, enum, or union
//
pub struct Foo3<#[inline] T>(PhantomData<T>);
//~^ ERROR attribute should be applied to function or closure
pub struct Bar3<#[cold] T>(PhantomData<T>);
//~^ ERROR attribute should be applied to a function
//~| WARN this was previously accepted
pub struct Baz3<#[repr(C)] T>(PhantomData<T>);
//~^ ERROR attribute should be applied to a struct, enum, or union

fn main() {}

// test for ICE "no entry found for key" in generics_of.rs #113017

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub fn String<V>(elem)
//~^ ERROR expected one of `:`, `@`, or `|`, found `)`
where
    V: 'a,
    //~^ ERROR use of undeclared lifetime name `'a`
    for<const N: usize = { || {}}> V: 'a,
    //~^ ERROR use of undeclared lifetime name `'a`
    //~^^ ERROR only lifetime parameters can be used in this context
    //~^^^ ERROR defaults for generic parameters are not allowed in `for<...>` binders
    for<C2: , R2, R3: > <&str as IntoIterator>::Item: 'static,
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    //~^^ ERROR only lifetime parameters can be used in this context
{}

pub fn main() {}

//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Loop = Loop; //[classic]~ ERROR overflow evaluating the requirement

impl Loop {} //~ ERROR overflow evaluating the requirement

type Poly0<T> = Poly1<(T,)>;
//[classic]~^ ERROR overflow evaluating the requirement
//[next]~^^ ERROR type parameter `T` is never used
type Poly1<T> = Poly0<(T,)>;
//[classic]~^ ERROR overflow evaluating the requirement
//[next]~^^ ERROR type parameter `T` is never used

impl Poly0<()> {} //~ ERROR overflow evaluating the requirement

fn main() {}

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Loop = Loop; //[current]~ ERROR overflow normalizing the type alias `Loop`

impl Loop {}
//[current]~^ ERROR overflow normalizing the type alias `Loop`
//[next]~^^ ERROR overflow evaluating the requirement `Loop == _`

type Poly0<T> = Poly1<(T,)>;
//[current]~^ ERROR overflow normalizing the type alias `Poly0<(((((((...,),),),),),),)>`
//[next]~^^ ERROR type parameter `T` is only used recursively
type Poly1<T> = Poly0<(T,)>;
//[current]~^ ERROR  overflow normalizing the type alias `Poly1<(((((((...,),),),),),),)>`
//[next]~^^ ERROR type parameter `T` is only used recursively

impl Poly0<()> {}
//[current]~^ ERROR overflow normalizing the type alias `Poly1<(((((((...,),),),),),),)>`
//[next]~^^ ERROR overflow evaluating the requirement `Poly0<()> == _`

fn main() {}

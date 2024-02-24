//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Loop = Loop; //[classic]~ ERROR overflow normalizing the type alias `Loop`

impl Loop {}
//[classic]~^ ERROR overflow normalizing the type alias `Loop`
//[next]~^^ ERROR overflow evaluating the requirement `Loop == _`

type Poly0<T> = Poly1<(T,)>;
//[classic]~^ ERROR overflow normalizing the type alias `Poly0<(((((((...,),),),),),),)>`
//[next]~^^ ERROR type parameter `T` is never used
type Poly1<T> = Poly0<(T,)>;
//[classic]~^ ERROR  overflow normalizing the type alias `Poly1<(((((((...,),),),),),),)>`
//[next]~^^ ERROR type parameter `T` is never used

impl Poly0<()> {}
//[classic]~^ ERROR overflow normalizing the type alias `Poly1<(((((((...,),),),),),),)>`
//[next]~^^ ERROR overflow evaluating the requirement `Poly0<()> == _`

fn main() {}

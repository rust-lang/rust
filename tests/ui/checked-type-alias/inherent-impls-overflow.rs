//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(checked_type_aliases)]
#![allow(incomplete_features)]

type Loop = Loop;
//[current]~^ ERROR overflow normalizing the type alias `Loop`
//[next]~^^ ERROR overflow evaluating the requirement `Loop == _`

impl Loop {}
//[current]~^ ERROR overflow normalizing the type alias `Loop`
//[next]~^^ ERROR overflow evaluating the requirement `Loop == _`
//[next]~| ERROR overflow evaluating the requirement `Loop == _`

type Poly0<T> = Poly1<(T,)>;
//[current]~^ ERROR overflow normalizing the type alias `Poly0<(((((((_,),),),),),),)>`
//[next]~^^ ERROR overflow evaluating the requirement
type Poly1<T> = Poly0<(T,)>;
//[current]~^ ERROR  overflow normalizing the type alias `Poly1<(((((((_,),),),),),),)>`
//[next]~^^ ERROR overflow evaluating the requirement

impl Poly0<()> {}
//[current]~^ ERROR overflow normalizing the type alias `Poly1<(((((((_,),),),),),),)>`
//[next]~^^ ERROR overflow evaluating the requirement `Poly0<()> == _`
//[next]~| ERROR overflow evaluating the requirement

fn main() {}

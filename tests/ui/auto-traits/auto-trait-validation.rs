#![feature(auto_traits)]
#![allow(dead_code)]

//@ run-rustfix

auto trait Generic<T> {}
//~^ ERROR auto traits cannot have generic parameters [E0567]
auto trait Bound : Copy {}
//~^ ERROR auto traits cannot have super traits or lifetime bounds [E0568]
auto trait LifetimeBound : 'static {}
//~^ ERROR auto traits cannot have super traits or lifetime bounds [E0568]
auto trait MyTrait { fn foo() {} }
//~^ ERROR auto traits cannot have associated items [E0380]
fn main() {}

#![feature(auto_traits)]
#![allow(dead_code)]

//@ run-rustfix

auto trait Generic<T> {}
//~^ auto traits cannot have generic parameters [E0567]
auto trait Bound : Copy {}
//~^ auto traits cannot have super traits or lifetime bounds [E0568]
auto trait LifetimeBound : 'static {}
//~^ auto traits cannot have super traits or lifetime bounds [E0568]
auto trait MyTrait { fn foo() {} }
//~^ auto traits cannot have associated items [E0380]
fn main() {}

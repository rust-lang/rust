#![feature(auto_traits)]

auto trait Generic<T> {}
//~^ auto traits cannot have generic parameters [E0567]
auto trait Bound : Copy {}
//~^ auto traits cannot have super traits [E0568]
auto trait MyTrait { fn foo() {} }
//~^ auto traits cannot have methods or associated items [E0380]
fn main() {}

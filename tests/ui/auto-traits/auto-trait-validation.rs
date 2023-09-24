#![feature(rustc_attrs)]

// run-rustfix

#[rustc_auto_trait]
trait Generic<T> {}
//~^ auto traits cannot have generic parameters [E0567]
#[rustc_auto_trait]
trait Bound : Copy {}
//~^ auto traits cannot have super traits or lifetime bounds [E0568]
#[rustc_auto_trait]
trait LifetimeBound : 'static {}
//~^ auto traits cannot have super traits or lifetime bounds [E0568]
#[rustc_auto_trait]
trait MyTrait { fn foo() {} }
//~^ auto traits cannot have associated items [E0380]

fn main() {}

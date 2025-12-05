#![feature(trait_alias)]
trait B = Fn() -> Self;
type D = &'static dyn B;
//~^ ERROR E0411

fn a() -> D {
    unreachable!();
}

fn main() {
    _ = a();
}

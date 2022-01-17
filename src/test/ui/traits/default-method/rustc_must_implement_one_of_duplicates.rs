#![feature(rustc_attrs)]

#[rustc_must_implement_one_of(a, a)]
//~^ Functions names are duplicated
trait Trait {
    fn a() {}
}

#[rustc_must_implement_one_of(b, a, a, c, b, c)]
//~^ Functions names are duplicated
//~| Functions names are duplicated
//~| Functions names are duplicated
trait Trait1 {
    fn a() {}
    fn b() {}
    fn c() {}
}

fn main() {}

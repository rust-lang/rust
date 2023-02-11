#![feature(rustc_attrs)]

#[rustc_must_implement_one_of(a, a)]
//~^ functions names are duplicated
trait Trait {
    fn a() {}
}

#[rustc_must_implement_one_of(b, a, a, c, b, c)]
//~^ functions names are duplicated
//~| functions names are duplicated
//~| functions names are duplicated
trait Trait1 {
    fn a() {}
    fn b() {}
    fn c() {}
}

fn main() {}

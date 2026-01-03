#![feature(supertrait_item_shadowing)]
#![deny(shadowing_supertrait_items)]

trait SuperSuper {
    fn method();
}

trait Super: SuperSuper {
    fn method();
    //~^ ERROR trait item `method` from `Super` shadows identically named item
}

trait Sub: Super {
    fn method();
    //~^ ERROR trait item `method` from `Sub` shadows identically named item
}

fn main() {}

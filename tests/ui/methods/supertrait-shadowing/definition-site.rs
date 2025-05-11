#![feature(supertrait_item_shadowing)]
#![deny(supertrait_item_shadowing_definition)]

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

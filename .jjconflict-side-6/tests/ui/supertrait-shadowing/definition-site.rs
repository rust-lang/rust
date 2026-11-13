#![feature(supertrait_item_shadowing)]
#![deny(shadowing_supertrait_items)]

trait SuperSuper {
    fn method();
    const CONST: i32;
    type Assoc;
}

trait Super: SuperSuper {
    fn method();
    //~^ ERROR trait item `method` from `Super` shadows identically named item
    const CONST: i32;
    //~^ ERROR trait item `CONST` from `Super` shadows identically named item
    type Assoc;
    //~^ ERROR trait item `Assoc` from `Super` shadows identically named item
}

trait Sub: Super {
    fn method();
    //~^ ERROR trait item `method` from `Sub` shadows identically named item
    const CONST: i32;
    //~^ ERROR trait item `CONST` from `Sub` shadows identically named item
    type Assoc;
    //~^ ERROR trait item `Assoc` from `Sub` shadows identically named item
}

fn main() {}

#![allow(bare_trait_objects)]

trait Trait {}

fn f<'a, T: Trait + ('a)>() {} //~ ERROR parenthesized lifetime bounds are not supported

fn check<'a>() {
    let _: Box<Trait + ('a)>; //~ ERROR parenthesized lifetime bounds are not supported
    let _: Box<('a) + Trait>;
    //~^ ERROR expected type, found `'a`
    //~| ERROR expected `:`, found `)`
    //~| ERROR chained comparison operators require parentheses
}

fn main() {}

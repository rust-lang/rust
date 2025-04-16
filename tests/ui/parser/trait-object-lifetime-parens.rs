//@ revisions: e2015 e2021
//@[e2015] edition: 2015
//@[e2021] edition: 2021

#![cfg_attr(e2015, allow(bare_trait_objects))]

trait Trait {}

fn f<'a, T: Trait + ('a)>() {} //~ ERROR parenthesized lifetime bounds are not supported

fn check<'a>() {
    let _: Box<Trait + ('a)>; //~ ERROR parenthesized lifetime bounds are not supported
    //[e2021]~^ ERROR expected a type, found a trait
    // FIXME: It'd be great if we could suggest removing the parentheses here too.
    //[e2015]~v ERROR lifetimes must be followed by `+` to form a trait object type
    let _: Box<('a) + Trait>;
    //[e2021]~^ ERROR expected type, found lifetime
    //[e2021]~| ERROR expected a path on the left-hand side of `+`
}

fn main() {}

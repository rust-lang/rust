#![allow(bare_trait_objects)]

trait Trait {}

fn f<'a, T: Trait + ('a)>() {} //~ ERROR parenthesized lifetime bounds are not supported

fn check<'a>() {
    let _: Box<Trait + ('a)>; //~ ERROR parenthesized lifetime bounds are not supported
    // FIXME: It'd be great if we could add suggestion to the following case.
    let _: Box<('a) + Trait>; //~ ERROR lifetime in trait object type must be followed by `+`
}

fn main() {}

// Can't put mut in non-ident pattern

//@ edition:2018
//@ dont-require-annotations: HELP

#![feature(box_patterns)]
#![allow(warnings)]

pub fn main() {
    let mut _ = 0; //~ ERROR `mut` must be followed by a named binding
    let mut (_, _) = (0, 0); //~ ERROR `mut` must be followed by a named binding

    let mut (x @ y) = 0; //~ ERROR `mut` must be attached to each individual binding

    let mut mut x = 0;
    //~^ ERROR `mut` on a binding may not be repeated
    //~| HELP remove the additional `mut`s

    let mut mut mut mut mut x = 0;
    //~^ ERROR `mut` on a binding may not be repeated
    //~| HELP remove the additional `mut`s

    struct Foo { x: isize }
    let mut Foo { x: x } = Foo { x: 3 };
    //~^ ERROR `mut` must be attached to each individual binding
    //~| HELP add `mut` to each binding

    let mut Foo { x } = Foo { x: 3 };
    //~^ ERROR `mut` must be attached to each individual binding
    //~| HELP add `mut` to each binding

    struct r#yield(u8, u8);
    let mut mut yield(become, await) = r#yield(0, 0);
    //~^ ERROR `mut` on a binding may not be repeated
    //~| ERROR `mut` must be followed by a named binding
    //~| ERROR expected identifier, found reserved keyword `yield`
    //~| ERROR expected identifier, found reserved keyword `become`
    //~| ERROR expected identifier, found keyword `await`

    struct W<T, U>(T, U);
    struct B { f: Box<u8> }
    let mut W(mut a, W(b, W(ref c, W(d, B { box f }))))
    //~^ ERROR `mut` must be attached to each individual binding
        = W(0, W(1, W(2, W(3, B { f: Box::new(4u8) }))));

    // Make sure we don't accidentally allow `mut $p` where `$p:pat`.
    macro_rules! foo {
        ($p:pat) => {
            let mut $p = 0; //~ ERROR expected identifier, found metavariable
        }
    }
    foo!(x);
}

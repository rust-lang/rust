#![feature(unboxed_closures)]

trait Zero { fn dummy(&self); }

fn foo1(_: dyn Zero()) {
    //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
    //~| ERROR associated type `Output` not found for `Zero`
}

fn foo2(_: dyn Zero<usize>) {
    //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
}

fn foo3(_: dyn Zero <   usize   >) {
    //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
}

fn foo4(_: dyn Zero(usize)) {
    //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
    //~| ERROR associated type `Output` not found for `Zero`
}

fn foo5(_: dyn Zero (   usize   )) {
    //~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
    //~| ERROR associated type `Output` not found for `Zero`
}

fn main() { }

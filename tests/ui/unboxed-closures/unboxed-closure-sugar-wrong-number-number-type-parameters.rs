#![feature(unboxed_closures)]

trait Zero { fn dummy(&self); }

fn foo1(_: &dyn Zero()) {
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument
    //~| ERROR associated type `Output` not found for `Zero`
}

fn foo2(_: &dyn Zero<usize>) {
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument
}

fn foo3(_: &dyn Zero <   usize   >) {
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument
}

fn foo4(_: &dyn Zero(usize)) {
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument
    //~| ERROR associated type `Output` not found for `Zero`
}

fn foo5(_: &dyn Zero (   usize   )) {
    //~^ ERROR trait takes 0 generic arguments but 1 generic argument
    //~| ERROR associated type `Output` not found for `Zero`
}

fn main() { }

#![feature(fn_traits)]
#![feature(unboxed_closures)]

struct S;

impl Fn(u32) -> u32 for S {
    //~^ ERROR associated item constraints are not allowed here [E0229]
    //~| ERROR expected a `FnMut(u32)` closure, found `S`
    fn call(&self) -> u32 {
        //~^ ERROR method `call` has 1 parameter but the declaration in trait `call` has 2
        5
    }
}

fn main() {}

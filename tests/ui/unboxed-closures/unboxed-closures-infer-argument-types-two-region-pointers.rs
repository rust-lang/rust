#![feature(fn_traits)]

// That a closure whose expected argument types include two distinct
// bound regions.

use std::cell::Cell;

fn doit<T,F>(val: T, f: &F)
    where F : Fn(&Cell<&T>, &T)
{
    let x = Cell::new(&val);
    f.call((&x,&val))
}

pub fn main() {
    doit(0, &|x, y| {
        x.set(y);
        //~^ ERROR lifetime may not live long enough
    });
}

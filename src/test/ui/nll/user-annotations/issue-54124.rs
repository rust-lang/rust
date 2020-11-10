#![feature(nll)]

fn test<'a>() {
    let _:fn(&()) = |_:&'a ()| {}; //~ ERROR lifetime may not be long enough
    //~^ ERROR lifetime may not be long enough
}

fn main() {
    test();
}

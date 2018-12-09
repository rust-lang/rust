#![feature(nll)]

fn test<'a>() {
    let _:fn(&()) = |_:&'a ()| {}; //~ ERROR unsatisfied lifetime constraints
    //~^ ERROR unsatisfied lifetime constraints
}

fn main() {
    test();
}

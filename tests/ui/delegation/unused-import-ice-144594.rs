#![allow(incomplete_features)]
#![feature(fn_delegation)]

reuse a as b {
    //~^ ERROR cannot find function `a` in this scope [E0425]
    || {
        use std::ops::Add;
        x.add
        //~^ ERROR cannot find value `x` in this scope [E0425]
    }
}

fn main() {}

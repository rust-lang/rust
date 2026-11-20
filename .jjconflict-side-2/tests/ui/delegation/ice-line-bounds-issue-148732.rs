reuse a as b { //~ ERROR failed to resolve delegation callee
    //~^ ERROR cannot find function `a` in this scope
    //~| ERROR functions delegation is not yet fully implemented
    dbg!(b);
    //~^ ERROR: `fn() {b}` doesn't implement `Debug`
}

fn main() {}

reuse a as b {
    //~^ ERROR cannot find function `a` in this scope
    //~| ERROR functions delegation is not yet fully implemented
    dbg!(b);
    //~^ ERROR missing lifetime specifier
    //~| ERROR `fn() {b}` doesn't implement `Debug`
}

fn main() {}

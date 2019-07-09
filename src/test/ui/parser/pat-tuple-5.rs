fn main() {
    match (0, 1) {
        (pat ..) => {}
        //~^ ERROR `X..` range patterns are not supported
        //~| ERROR arbitrary expressions aren't allowed in patterns
        //~| ERROR cannot find value `pat` in this scope
        //~| ERROR exclusive range pattern syntax is experimental
        //~| ERROR only char and numeric types are allowed in range patterns
    }
}

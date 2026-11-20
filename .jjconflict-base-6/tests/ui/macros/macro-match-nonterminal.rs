macro_rules! test {
    ($a, $b) => {
        //~^ ERROR missing fragment
        //~| ERROR missing fragment
        ()
    };
}

fn main() {
    test!() //~ ERROR unexpected end of macro invocation
}

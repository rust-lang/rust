mod A {
    struct C;
}

fn main() {
    A::C = 1;
    //~^ ERROR: mismatched types
    //~| ERROR: unit struct `C` is private
}

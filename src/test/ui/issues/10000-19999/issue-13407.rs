mod A {
    struct C;
}

fn main() {
    A::C = 1;
    //~^ ERROR: invalid left-hand side of assignment
    //~| ERROR: mismatched types
    //~| ERROR: struct `C` is private
}

enum Test {
    DivZero = 1/0,
    //~^ NOTE attempt to divide `1_isize` by zero
    //~| ERROR evaluation of constant value failed
    RemZero = 1%0,
    //~^ NOTE attempt to calculate the remainder of `1_isize` with a divisor of zero
    //~| ERROR evaluation of constant value failed
}

fn main() {}

enum Test {
    DivZero = 1/0,
    //~^ attempt to divide by zero
    //~| ERROR evaluation of constant value failed
    RemZero = 1%0,
    //~^ attempt to calculate the remainder with a divisor of zero
    //~| ERROR evaluation of constant value failed
}

fn main() {}

enum Test {
    DivZero = 1 / 0,
    //~^ ERROR attempt to divide `1_isize` by zero
    RemZero = 1 % 0,
    //~^ ERROR attempt to calculate the remainder of `1_isize` with a divisor of zero
}

fn main() {}

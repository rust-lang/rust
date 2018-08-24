enum Test {
    DivZero = 1/0,
    //~^ attempt to divide by zero
    //~| ERROR could not evaluate enum discriminant
    //~| ERROR this expression will panic at runtime
    RemZero = 1%0,
    //~^ attempt to calculate the remainder with a divisor of zero
    //~| ERROR could not evaluate enum discriminant
    //~| ERROR this expression will panic at runtime
}

fn main() {}

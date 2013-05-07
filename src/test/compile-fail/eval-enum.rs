enum test {
    div_zero = 1/0, //~ERROR expected constant: attempted to divide by zero
    rem_zero = 1%0  //~ERROR expected constant: attempted remainder with a divisor of zero
}

fn main() {}

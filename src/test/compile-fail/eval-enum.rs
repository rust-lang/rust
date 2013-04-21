enum test {
    quot_zero = 1/0, //~ERROR expected constant: quotient zero
    rem_zero = 1%0  //~ERROR expected constant: remainder zero
}

fn main() {}

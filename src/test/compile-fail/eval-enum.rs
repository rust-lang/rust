enum test {
    div_zero = 1/0, //~ERROR expected constant: divide by zero
    rem_zero = 1%0  //~ERROR expected constant: modulo zero
}

fn main() {}

enum Enum {
    X = (1 << 500), //~ ERROR E0080
    //~| NOTE attempt to shift left by `500_i32`, which would overflow
    Y = (1 / 0) //~ ERROR E0080
                //~| NOTE attempt to divide `1_isize` by zero
}

fn main() {
}

enum Enum {
    X = (1 << 500), //~ ERROR E0080
    //~| shift left with overflow
    Y = (1 / 0) //~ ERROR E0080
}

fn main() {
}

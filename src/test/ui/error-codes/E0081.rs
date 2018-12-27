enum Enum {
    P = 3,
    X = 3,
    //~^ ERROR discriminant value `3` already exists
    Y = 5
}

fn main() {
}

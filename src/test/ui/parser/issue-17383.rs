enum X {
    A = 3,
    //~^ ERROR custom discriminant values are not allowed in enums with tuple or struct variants
    B(usize)
}

fn main() {}

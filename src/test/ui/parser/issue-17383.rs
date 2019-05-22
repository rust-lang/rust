enum X {
    A = 3,
    //~^ ERROR custom discriminant values are not allowed in enums with fields
    B(usize)
}

fn main() {}

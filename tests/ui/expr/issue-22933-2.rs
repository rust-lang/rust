enum Delicious {
    Pie      = 0x1,
    Apple    = 0x2,
    ApplePie = Delicious::Apple as isize | Delicious::PIE as isize,
    //~^ ERROR no variant, associated function, or constant named `PIE` found
}

fn main() {}

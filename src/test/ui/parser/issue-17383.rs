enum X {
    A = 3,
    //~^ ERROR discriminator values can only be used with a field-less enum
    B(usize)
}

fn main() {}

enum Enum<T: Trait> {
    X = Trait::Number,
    //~^ ERROR mismatched types
    //~| expected isize, found i32
}

trait Trait {
    const Number: i32 = 1;
}

fn main() {}

enum Enum<T: Trait> {
    //~^ ERROR `T` is never used
    //~| NOTE unused type parameter
    X = Trait::Number,
    //~^ ERROR mismatched types
    //~| NOTE expected `isize`, found `i32`
}

trait Trait {
    const Number: i32 = 1;
}

fn main() {}

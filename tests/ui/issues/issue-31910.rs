enum Enum<T: Trait> {
    //~^ ERROR `T` is never used
    X = Trait::Number,
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected `isize`, found `i32`
}

trait Trait {
    const Number: i32 = 1;
}

fn main() {}

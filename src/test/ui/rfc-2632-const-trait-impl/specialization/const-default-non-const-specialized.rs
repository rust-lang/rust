// Tests that a const default trait impl can be specialized by a non-const trait
// impl, but that the specializing impl cannot be used in a const context.

#![feature(const_trait_impl)]
#![feature(min_specialization)]

trait Value {
    fn value() -> u32;
}

const fn get_value<T: ~const Value>() -> u32 {
    T::value()
    //~^ ERROR any use of this value will cause an error [const_err]
    //~| WARNING this was previously accepted
}

impl<T> const Value for T {
    default fn value() -> u32 {
        0
    }
}

struct FortyTwo;

impl Value for FortyTwo {
    fn value() -> u32 {
        println!("You can't do that (constly)");
        42
    }
}

const ZERO: u32 = get_value::<()>();

const FORTY_TWO: u32 =
    get_value::<FortyTwo>(); // This is the line that causes the error, but it gets reported above

fn main() {}

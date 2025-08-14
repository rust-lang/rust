// Make sure we don't ICE when evaluating a trait whose impl has a bad signature.

#![feature(const_trait_impl)]

#[const_trait]
trait Value {
    fn value() -> u32;
}

const fn get_value<T: [const] Value>() -> u32 {
    T::value()
}

struct FortyTwo;

impl const Value for FortyTwo {
    fn value() -> i64 {
        //~^ ERROR method `value` has an incompatible type for trait
        42
    }
}

const FORTY_TWO: u32 = get_value::<FortyTwo>();

fn main() {
    assert_eq!(FORTY_TWO, 42);
}

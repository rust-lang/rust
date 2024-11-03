//@ known-bug: #112623

#![feature(const_trait_impl)]

#[const_trait]
trait Value {
    fn value() -> u32;
}

const fn get_value<T: ~const Value>() -> u32 {
    T::value()
}

struct FortyTwo;

impl const Value for FortyTwo {
    fn value() -> i64 {
        42
    }
}

const FORTY_TWO: u32 = get_value::<FortyTwo>();

fn main() {
    assert_eq!(FORTY_TWO, 42);
}

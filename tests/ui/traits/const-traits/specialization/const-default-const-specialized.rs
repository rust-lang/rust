// Tests that a const default trait impl can be specialized by another const
// trait impl and that the specializing impl will be used during const-eval.

//@ run-pass

#![feature(const_trait_impl)]
#![feature(min_specialization)]

#[const_trait]
trait Value {
    fn value() -> u32;
}

const fn get_value<T: [const] Value>() -> u32 {
    T::value()
}

impl<T> const Value for T {
    default fn value() -> u32 {
        0
    }
}

struct FortyTwo;

impl const Value for FortyTwo {
    fn value() -> u32 {
        42
    }
}

const ZERO: u32 = get_value::<()>();

const FORTY_TWO: u32 = get_value::<FortyTwo>();

fn main() {
    assert_eq!(ZERO, 0);
    assert_eq!(FORTY_TWO, 42);
}

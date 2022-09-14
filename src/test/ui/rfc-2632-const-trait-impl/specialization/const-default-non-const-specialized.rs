// Tests that a const default trait impl cannot be specialized by a non-const
// trait impl.

#![feature(const_trait_impl)]
#![feature(min_specialization)]

trait Value {
    fn value() -> u32;
}

impl<T> const Value for T {
    default fn value() -> u32 {
        0
    }
}

struct FortyTwo;

impl Value for FortyTwo { //~ ERROR cannot specialize on const impl with non-const impl
    fn value() -> u32 {
        println!("You can't do that (constly)");
        42
    }
}

fn main() {}

// Previously, enums with non-int discrs ICEd during CTFE of promoteds.

enum A {
//~^ ERROR `#[repr(inttype)]` must be specified
    V1(isize) = 1..=10,
    //~^ ERROR mismatched types
    V0 = 1..=10,
    //~^ ERROR mismatched types
}

const B: &'static [A] = &[A::V0, A::V1(111)];

fn main() {}

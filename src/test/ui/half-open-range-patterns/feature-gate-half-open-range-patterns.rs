#![feature(exclusive_range_pattern)]

fn main() {}

#[cfg(FALSE)]
fn foo() {
    if let ..=5 = 0 {}
    //~^ ERROR half-open range patterns are unstable
    if let ...5 = 0 {}
    //~^ ERROR half-open range patterns are unstable
    if let ..5 = 0 {}
    //~^ ERROR half-open range patterns are unstable
    if let 5.. = 0 {}
    //~^ ERROR half-open range patterns are unstable
    if let 5..= = 0 {}
    //~^ ERROR half-open range patterns are unstable
    //~| ERROR inclusive range with no end
    if let 5... = 0 {}
    //~^ ERROR half-open range patterns are unstable
    //~| ERROR inclusive range with no end
}

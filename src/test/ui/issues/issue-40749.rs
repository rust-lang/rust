fn main() {
    [0; ..10];
    //~^ ERROR mismatched types
    //~| expected type `usize`
    //~| found struct `std::ops::RangeTo<{integer}>`
}

fn main() {
    [(); & { loop { continue } } ]; //~ ERROR mismatched types

    [(); loop { break }]; //~ ERROR mismatched types

    [(); {while true {break}; 0}];
    //~^ WARN denote infinite loops with

    [(); { for _ in 0usize.. {}; 0}];
    //~^ ERROR `std::ops::RangeFrom<usize>: const Iterator` is not satisfied
    //~| ERROR `std::ops::RangeFrom<usize>: const Iterator` is not satisfied
}

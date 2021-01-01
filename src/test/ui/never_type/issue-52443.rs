fn main() {
    [(); & { loop { continue } } ]; //~ ERROR mismatched types

    [(); loop { break }]; //~ ERROR mismatched types

    [(); {while true {break}; 0}];
    //~^ WARN denote infinite loops with

    [(); { for _ in 0usize.. {}; 0}];
    //~^ ERROR `for` is not allowed in a `const`
    //~| ERROR calls in constants are limited to constant functions
    //~| ERROR mutable references are not allowed in constants
    //~| ERROR calls in constants are limited to constant functions
}

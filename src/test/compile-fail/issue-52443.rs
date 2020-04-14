fn main() {
    [(); & { loop { continue } } ]; //~ ERROR mismatched types
    //~^ ERROR `loop` is not allowed in a `const`
    [(); loop { break }]; //~ ERROR mismatched types
    //~^ ERROR `loop` is not allowed in a `const`
    [(); {while true {break}; 0}];
    //~^ ERROR `while` is not allowed in a `const`
    //~| WARN denote infinite loops with
    [(); { for _ in 0usize.. {}; 0}];
    //~^ ERROR `for` is not allowed in a `const`
    //~| ERROR calls in constants are limited to constant functions
    //~| ERROR references in constants may only refer to immutable values
    //~| ERROR calls in constants are limited to constant functions
    //~| ERROR constant contains unimplemented expression type
    //~| ERROR evaluation of constant value failed
}

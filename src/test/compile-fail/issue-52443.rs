fn main() {
    [(); & { loop { continue } } ]; //~ ERROR mismatched types
    //~^ ERROR `loop` is not allowed in a `const`
    [(); loop { break }]; //~ ERROR mismatched types
    //~^ ERROR `loop` is not allowed in a `const`
    [(); {while true {break}; 0}];
    //~^ ERROR constant contains unimplemented expression type
    //~| ERROR constant contains unimplemented expression type
    //~| ERROR `while` is not allowed in a `const`
    //~| WARN denote infinite loops with
    [(); { for _ in 0usize.. {}; 0}];
    //~^ ERROR calls in constants are limited to constant functions
    //~| ERROR `for` is not allowed in a `const`
    //~| ERROR references in constants may only refer to immutable values
    //~| ERROR constant contains unimplemented expression type
    //~| ERROR constant contains unimplemented expression type
    //~| ERROR evaluation of constant value failed
}

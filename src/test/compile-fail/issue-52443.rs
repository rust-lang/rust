fn main() {
    [(); & { loop { continue } } ]; //~ ERROR mismatched types
    [(); loop { break }]; //~ ERROR mismatched types
    [(); {while true {break}; 0}];
    //~^ ERROR constant contains unimplemented expression type
    //~| ERROR constant contains unimplemented expression type
    //~| WARN denote infinite loops with
    [(); { for _ in 0usize.. {}; 0}];
    //~^ ERROR calls in constants are limited to constant functions
    //~| ERROR references in constants may only refer to immutable values
    //~| ERROR constant contains unimplemented expression type
    //~| ERROR constant contains unimplemented expression type
    //~| ERROR evaluation of constant value failed
}

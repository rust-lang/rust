fn main() {
    [(); & { loop { continue } } ]; //~ ERROR mismatched types
    [(); loop { break }]; //~ ERROR mismatched types
    [(); {while true {break}; 0}]; //~ ERROR constant contains unimplemented expression type
    [(); { for _ in 0usize.. {}; 0}]; //~ ERROR calls in constants are limited to constant functions
    //~^ ERROR constant contains unimplemented expression type
    //~| ERROR constant contains unimplemented expression type
    //~| ERROR evaluation of constant value failed
}

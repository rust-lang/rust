fn main() {
    |_: [_; continue]| {}; //~ ERROR: `continue` outside of a loop

    while |_: [_; continue]| {} {} //~ ERROR: `continue` outside of a loop
    //~^ ERROR mismatched types

    while |_: [_; break]| {} {} //~ ERROR: `break` outside of a loop
    //~^ ERROR mismatched types
}

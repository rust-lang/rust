fn main() {
    loop {
        |_: [_; break]| {} //~ ERROR: `break` outside of a loop
        //~^ ERROR mismatched types
    }

    loop {
        |_: [_; continue]| {} //~ ERROR: `continue` outside of a loop
        //~^ ERROR mismatched types
    }
}

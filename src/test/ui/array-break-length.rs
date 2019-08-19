fn main() {
    loop {
        |_: [_; break]| {} //~ ERROR: `break` outside of loop
        //~^ ERROR mismatched types
    }

    loop {
        |_: [_; continue]| {} //~ ERROR: `continue` outside of loop
        //~^ ERROR mismatched types
    }
}

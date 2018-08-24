fn main() {
    loop {
        |_: [_; break]| {} //~ ERROR: `break` outside of loop
    }

    loop {
        |_: [_; continue]| {} //~ ERROR: `continue` outside of loop
    }
}

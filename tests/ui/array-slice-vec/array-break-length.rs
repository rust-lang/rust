fn main() {
    loop {
        |_: [_; break]| {} //~ ERROR: `break` outside of a loop
    }

    loop {
        |_: [_; continue]| {} //~ ERROR: `continue` outside of a loop
    }
}

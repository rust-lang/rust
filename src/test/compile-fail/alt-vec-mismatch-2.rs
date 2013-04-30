fn main() {
    match () {
        [()] => { } //~ ERROR mismatched types: expected `()` but found a vector pattern
    }
}

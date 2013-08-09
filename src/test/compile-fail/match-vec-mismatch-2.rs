fn main() {
    match () {
        [()] => { } //~ ERROR mismatched types: expected `()`, found a vector pattern
    }
}

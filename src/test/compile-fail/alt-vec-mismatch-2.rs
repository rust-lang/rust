fn main() {
    match () {
        [()] => { } //~ ERROR mismatched type: expected `()` but found a vector pattern
    }
}

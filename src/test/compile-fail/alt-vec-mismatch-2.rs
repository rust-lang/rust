fn main() {
    match () {
        [()] => { } //~ ERROR mismatched type: expected `()` but found vector
    }
}

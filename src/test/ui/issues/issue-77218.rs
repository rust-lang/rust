fn main() {
    let value = [7u8];
    while Some(0) = value.get(0) { //~ ERROR mismatched types
        //~^ NOTE expected `bool`, found `()`
        //~| HELP you might have meant to use pattern matching
    }
}

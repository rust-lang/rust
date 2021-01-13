fn main() {
    let value = [7u8];
    while Some(0) = value.get(0) { //~ ERROR destructuring assignments are unstable
        //~| ERROR invalid left-hand side of assignment
        //~| ERROR mismatched types
        //~| ERROR mismatched types

        // FIXME The following diagnostic should also be emitted
        // HELP you might have meant to use pattern matching
    }
}

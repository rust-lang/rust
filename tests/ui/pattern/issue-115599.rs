// Check that we do not ICE when matching uninitialized constant with constant pattern

const CONST_STRING: String = String::new();

fn main() {
    let empty_str = String::from("");
    if let CONST_STRING = empty_str {}
    //~^ ERROR constant of non-structural type `String` in a pattern
}

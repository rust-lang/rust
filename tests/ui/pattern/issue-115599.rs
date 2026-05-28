const CONST_STRING: String = String::new();

fn main() {
    let empty_str = String::from("");
    if let CONST_STRING = empty_str {}
    //~^ ERROR constant of non-structural type `Vec<u8>` in a pattern
}

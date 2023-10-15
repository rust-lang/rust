const CONST_STRING: String = String::new();

fn main() {
    let empty_str = String::from("");
    if let CONST_STRING = empty_str {}
    //~^ ERROR to use a constant of type `Vec<u8>` in a pattern, `Vec<u8>` must be annotated with `#[derive(PartialEq, Eq)]`
}

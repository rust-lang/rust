fn main() {
    let baz = ().foo(); //~ ERROR no method named `foo` found for type `()` in the current scope
    <i32 as std::str::FromStr>::from_str(&baz); // No complaints about `str` being unsized
}

fn main() {
    let baz = ().foo(); //~ ERROR no method named `foo` found
    <i32 as std::str::FromStr>::from_str(&baz); // No complaints about `str` being unsized
}

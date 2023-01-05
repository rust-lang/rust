fn main() {}
fn foo(mut s: String) -> String {
    s.push_str("asdf") //~ ERROR mismatched types
}

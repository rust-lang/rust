// compile-flags: -Z parse-only


pub fn main() {
    br##"a"#;  //~ unterminated raw string
}

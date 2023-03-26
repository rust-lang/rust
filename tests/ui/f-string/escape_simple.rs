// run-pass
#![feature(f_strings)]

pub fn main() {
    let text = f"a{{b\{c}}d\}e";
    assert_eq!(text, "a{b{c}d}e");
}

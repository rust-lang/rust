// run-pass
#![feature(f_strings)]

pub fn main() {
    let a = "a".to_string();
    let b = "b".to_string();
    assert_eq!(f"{a}, {b}", "a, b");
}

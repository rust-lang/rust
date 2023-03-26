// run-pass
#![feature(f_strings)]

pub fn main() {
    assert_eq!(f"{ f"{ "a" }" + &f"{ "b" }" + "c" }", "abc");
}

// run-pass
#![feature(f_strings)]

pub fn main() {
    let a = 2;
    let b = 4;
    let c = f"a ({a}) + b ({b}) = {a + b}";
    assert_eq!(c, "a (2) + b (4) = 6");
}

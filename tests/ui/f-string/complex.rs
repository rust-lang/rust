// run-pass
#![feature(f_strings)]

pub fn main() {
    let b = 1;
    let d = 2;
    let e = 3;
    let g = 4;
    let result = f"a = { f"b" + { f"c{d + e}f" + g }}h";
    assert_eq!(result, "a = bc5 TODO");
}

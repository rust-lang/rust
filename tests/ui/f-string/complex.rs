// run-pass
#![feature(f_strings)]

pub fn main() {
    let d = 2;
    let e = 3;
    let g = 4;
    let result = f"a = { {f"b" + &{ (); f"c{d + e}f" + &g.to_string() }}}h";
    assert_eq!(result, "a = bc5f4h");
}

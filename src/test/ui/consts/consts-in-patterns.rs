// run-pass

const FOO: isize = 10;
const BAR: isize = 3;

const fn foo() -> isize { 4 }
const BOO: isize = foo();

pub fn main() {
    let x: isize = 3;
    let y = match x {
        FOO => 1,
        BAR => 2,
        BOO => 4,
        _ => 3
    };
    assert_eq!(y, 2);
}

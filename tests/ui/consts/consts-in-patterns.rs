//@ run-pass

const FOO: isize = 10;
const BAR: isize = 3;
const ZST: &() = unsafe { std::mem::transmute(1usize) };
const ZST_ARR: &[u8; 0] = unsafe { std::mem::transmute(1usize) };

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
    let z = match &() {
        ZST => 9,
    };
    assert_eq!(z, 9);
    let z = match b"" {
        ZST_ARR => 10,
    };
    assert_eq!(z, 10);
}

//@ revisions: noopt opt
//@ [noopt]compile-flags: -C opt-level=0
//@ [opt]compile-flags: -O
//@ build-fail

pub fn shr() -> u8 {
    let mut c: u8 = 10;
    c >>= 1;
    0xff + c //~ ERROR this arithmetic operation will overflow
}

pub fn shl() -> u8 {
    let mut c: u8 = 10;
    c <<= 1;
    0xff + c //~ ERROR this arithmetic operation will overflow
}

pub fn div() -> u8 {
    let mut c: u8 = 10;
    c /= 2;
    0xff + c //~ ERROR this arithmetic operation will overflow
}

fn main() {}

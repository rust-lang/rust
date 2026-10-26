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

pub fn chain() -> u8 {
    let mut c: u8 = 200;
    c >>= 1;
    c /= 1;
    c %= 200;
    0xff + c //~ ERROR this arithmetic operation will overflow
}

// Not known after the loop, so no error here.
pub fn in_loop(n: u32) -> u8 {
    let mut c: u8 = 200;
    let mut i = 0;
    while i < n {
        c >>= 1;
        i += 1;
    }
    c + 100
}

fn main() {}

// compile-flags: -Z threads=16
// build-pass

pub static GLOBAL: isize = 3;

static GLOBAL0: isize = 4;

pub static GLOBAL2: &'static isize = &GLOBAL0;

pub fn verify_same(a: &'static isize) {
    let a = a as *const isize as usize;
    let b = &GLOBAL as *const isize as usize;
    assert_eq!(a, b);
}

pub fn verify_same2(a: &'static isize) {
    let a = a as *const isize as usize;
    let b = GLOBAL2 as *const isize as usize;
    assert_eq!(a, b);
}

fn main() {}

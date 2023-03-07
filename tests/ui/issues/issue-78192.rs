// run-pass

#![allow(unused_assignments)]

fn main() {
    let a = 1u32;
    let b = 2u32;

    let mut c: *const u32 = &a;
    let d: &u32 = &b;

    let x = unsafe { &*c };
    c = d;
    let z = *x;

    assert_eq!(1, z);
}

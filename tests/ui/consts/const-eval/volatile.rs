//@ check-pass
#![feature(const_volatile)]

const _READ: () = unsafe {
    let x = 42i32;
    let y = (&x as *const i32).read_volatile();
    assert!(x == y);
};

const _WRITE: () = unsafe {
    let mut x = 42i32;
    (&mut x as *mut i32).write_volatile(13);
    assert!(x == 13);
};

fn main() {}

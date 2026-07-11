//@ run-pass

#![allow(dead_code)]
#![allow(unused_variables)]

// Drop works for union itself.

#[derive(Copy, Clone)]
struct S;

union U {
    a: u8
}

union W {
    a: S,
}

union Y {
    a: S,
}

static mut CHECK: u8 = 0;

fn increment_check() {
    unsafe {
        let check = &raw mut CHECK;
        check.write(check.read() + 1);
    }
}

fn check() -> u8 {
    unsafe { (&raw const CHECK).read() }
}

impl Drop for U {
    fn drop(&mut self) {
        increment_check();
    }
}

impl Drop for W {
    fn drop(&mut self) {
        increment_check();
    }
}

fn main() {
    assert_eq!(check(), 0);
    {
        let u = U { a: 1 };
    }
    assert_eq!(check(), 1); // 1, dtor of U is called
    {
        let w = W { a: S };
    }
    assert_eq!(check(), 2); // 2, dtor of W is called
    {
        let y = Y { a: S };
    }
    assert_eq!(check(), 2); // 2, Y has no dtor
    {
        let u2 = U { a: 1 };
        std::mem::forget(u2);
    }
    assert_eq!(check(), 2); // 2, dtor of U *not* called for u2
}

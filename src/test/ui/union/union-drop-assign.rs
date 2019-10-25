// run-pass
#![allow(unused_assignments)]
#![allow(unions_with_drop_fields)]

// Drop works for union itself.

#![feature(untagged_unions)]

struct S;

union U {
    a: S
}

impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}

impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}

static mut CHECK: u8 = 0;

fn main() {
    unsafe {
        let mut u = U { a: S };
        assert_eq!(CHECK, 0);
        u = U { a: S };
        assert_eq!(CHECK, 1); // union itself is assigned, union is dropped, field is not dropped
        u.a = S;
        assert_eq!(CHECK, 11); // union field is assigned, field is dropped
    }
}

// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unions_with_drop_fields)]

// Drop works for union itself.

#![feature(untagged_unions)]

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

impl Drop for W {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}

static mut CHECK: u8 = 0;

fn main() {
    unsafe {
        assert_eq!(CHECK, 0);
        {
            let u = U { a: 1 };
        }
        assert_eq!(CHECK, 1); // 1, dtor of U is called
        {
            let w = W { a: S };
        }
        assert_eq!(CHECK, 2); // 2, not 11, dtor of S is not called
        {
            let y = Y { a: S };
        }
        assert_eq!(CHECK, 2); // 2, not 12, dtor of S is not called
    }
}

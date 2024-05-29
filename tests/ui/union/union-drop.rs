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
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        {
            let u = U { a: 1 };
        }
        assert_eq!(CHECK, 1); // 1, dtor of U is called
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        {
            let w = W { a: S };
        }
        assert_eq!(CHECK, 2); // 2, dtor of W is called
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        {
            let y = Y { a: S };
        }
        assert_eq!(CHECK, 2); // 2, Y has no dtor
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        {
            let u2 = U { a: 1 };
            std::mem::forget(u2);
        }
        assert_eq!(CHECK, 2); // 2, dtor of U *not* called for u2
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    }
}

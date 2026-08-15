// Regression test for #149143.
// The compiler did not check for a coercion from intrinsics
// to fn ptrs in all possible code paths that could lead to such a coercion.
// This caused an ICE during a later sanity check.

use std::mem::transmute;

fn main() {
    unsafe {
        let f = if true { transmute } else { safe_transmute };
        //~^ ERROR `if` and `else` have incompatible type

        let _: i64 = f(5i64);
    }
    unsafe {
        let f = if true { safe_transmute } else { transmute };
        //~^ ERROR `if` and `else` have incompatible type

        let _: i64 = f(5i64);
    }
}

unsafe fn safe_transmute<A, B>(x: A) -> B {
    panic!()
}

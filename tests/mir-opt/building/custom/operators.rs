// skip-filecheck
//@ compile-flags: --crate-type=lib
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

// EMIT_MIR operators.f.built.after.mir
#[custom_mir(dialect = "built")]
pub fn f(a: i32, b: bool) -> i32 {
    mir! {
        {
            a = -a;
            b = !b;
            a = a + a;
            a = a - a;
            a = a * a;
            a = a / a;
            a = a % a;
            a = a ^ a;
            a = a & a;
            a = a << a;
            a = a >> a;
            b = a == a;
            b = a < a;
            b = a <= a;
            b = a >= a;
            b = a > a;
            let res = Checked(a + a);
            b = res.1;
            a = res.0;
            RET = a;
            Return()
        }
    }
}

// EMIT_MIR operators.g.runtime.after.mir
#[custom_mir(dialect = "runtime")]
pub fn g(p: *const i32, q: *const [i32]) {
    mir! {
        {
            let a = PtrMetadata(p);
            let b = PtrMetadata(q);
            Return()
        }
    }
}

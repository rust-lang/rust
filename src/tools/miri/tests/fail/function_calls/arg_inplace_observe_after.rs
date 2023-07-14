#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

pub struct S(i32);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let unit: ();
        let _observe: i32;
        {
            let non_copy = S(42);
            // This could change `non_copy` in-place
            Call(unit, after_call, change_arg(Move(non_copy)))
        }
        after_call = {
            // So now we must not be allowed to observe non-copy again.
            _observe = non_copy.0; //~ERROR: uninitialized
            Return()
        }

    }
}

pub fn change_arg(mut x: S) {
    x.0 = 0;
}

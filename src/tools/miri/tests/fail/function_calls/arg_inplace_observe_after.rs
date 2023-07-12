#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

pub struct S(i32);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    // FIXME: the span is not great (probably caused by custom MIR)
    mir! { //~ERROR: uninitialized
        let unit: ();
        {
            let non_copy = S(42);
            // This could change `non_copy` in-place
            Call(unit, after_call, change_arg(Move(non_copy)))
        }
        after_call = {
            // So now we must not be allowed to observe non-copy again.
            let _observe = non_copy.0;
            Return()
        }

    }
}

pub fn change_arg(mut x: S) {
    x.0 = 0;
}

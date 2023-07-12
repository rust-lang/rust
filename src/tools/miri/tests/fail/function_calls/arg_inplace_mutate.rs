//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

pub struct S(i32);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let unit: ();
        {
            let non_copy = S(42);
            let ptr = std::ptr::addr_of_mut!(non_copy);
            // Inside `callee`, the first argument and `*ptr` are basically
            // aliasing places!
            Call(unit, after_call, callee(Move(*ptr), ptr))
        }
        after_call = {
            Return()
        }

    }
}

pub fn callee(x: S, ptr: *mut S) {
    // With the setup above, if `x` is indeed moved in
    // (i.e. we actually just get a pointer to the underlying storage),
    // then writing to `ptr` will change the value stored in `x`!
    unsafe { ptr.write(S(0)) };
    //~[stack]^ ERROR: not granting access
    //~[tree]| ERROR: /write access .* forbidden/
    assert_eq!(x.0, 42);
}

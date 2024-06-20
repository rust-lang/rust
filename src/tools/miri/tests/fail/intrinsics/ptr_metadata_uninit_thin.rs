//@compile-flags: -Zmiri-disable-validation
#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// This disables validation and uses custom MIR hit exactly the UB in the intrinsic,
// rather than getting UB from the typed load or parameter passing.

#[custom_mir(dialect = "runtime")]
pub unsafe fn deref_meta(p: *const *const i32) -> () {
    mir! {
        {
            RET = PtrMetadata(*p); //~ ERROR: Undefined Behavior: using uninitialized data
            Return()
        }
    }
}

fn main() {
    // Even though the meta is the trivially-valid `()`, this is still UB

    let p = std::mem::MaybeUninit::<*const i32>::uninit();
    unsafe {
        let _meta = deref_meta(p.as_ptr());
    }
}

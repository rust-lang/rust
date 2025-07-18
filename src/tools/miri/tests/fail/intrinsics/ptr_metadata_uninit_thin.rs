//@compile-flags: -Zmiri-disable-validation
//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// This disables validation and uses custom MIR hit exactly the UB in the intrinsic,
// rather than getting UB from the typed load or parameter passing.

#[custom_mir(dialect = "runtime")]
pub unsafe fn deref_meta(p: *const *const i32) -> () {
    mir! {
        {
            RET = PtrMetadata(*p); //~ ERROR: /Undefined Behavior: .*, but memory is uninitialized/
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

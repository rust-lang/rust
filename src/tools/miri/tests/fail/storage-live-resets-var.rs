#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime")]
fn main() {
    mir! {
        let val: i32;
        let _val2: i32;
        {
            StorageLive(val);
            val = 42;
            StorageLive(val); // reset val to `uninit`
            _val2 = val; //~ERROR: uninitialized
            Return()
        }
    }
}

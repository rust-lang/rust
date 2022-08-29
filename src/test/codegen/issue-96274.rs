// min-llvm-version: 15.0
// compile-flags: -O

#![crate_type = "lib"]

use std::mem::MaybeUninit;

pub fn maybe_uninit() -> [MaybeUninit<u8>; 3000] {
    // CHECK-NOT: memset
    [MaybeUninit::uninit(); 3000]
}

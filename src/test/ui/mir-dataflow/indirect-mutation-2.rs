#![feature(core_intrinsics, rustc_attrs, const_raw_ptr_deref)]

use std::cell::Cell;
use std::intrinsics::rustc_peek;

struct Arr {
    inner: [Cell<i32>; 0],
    _peek: (),
}

// Zero-sized arrays are never mutable.
#[rustc_mir(rustc_peek_indirectly_mutable, stop_after_dataflow)]
pub fn zst(flag: bool) {
    {
        let arr: [i32; 0] = [];
        let s: &mut [i32] = &mut arr;
        unsafe { rustc_peek(arr) }; //~ ERROR rustc_peek: bit not set
    }

    {
        let arr: [Cell<i32>; 0] = [];
        let s: &[Cell<i32>] = &arr;
        unsafe { rustc_peek(arr) }; //~ ERROR rustc_peek: bit not set

        let ss = &s;
        unsafe { rustc_peek(arr) }; //~ ERROR rustc_peek: bit not set
    }
}

fn main() {}

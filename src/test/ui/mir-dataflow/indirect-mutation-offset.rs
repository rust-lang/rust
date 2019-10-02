// compile-flags: -Zunleash-the-miri-inside-of-you

#![feature(core_intrinsics, rustc_attrs, const_raw_ptr_deref)]

use std::cell::UnsafeCell;
use std::intrinsics::rustc_peek;

#[repr(C)]
struct PartialInteriorMut {
    zst: [i32; 0],
    cell: UnsafeCell<i32>,
}

#[rustc_mir(rustc_peek_indirectly_mutable,stop_after_dataflow)]
#[rustc_mir(borrowck_graphviz_postflow="indirect.dot")]
const BOO: i32 = {
    let x = PartialInteriorMut {
        zst: [],
        cell: UnsafeCell::new(0),
    };

    unsafe { rustc_peek(x) }; //~ ERROR rustc_peek: bit not set

    let p_zst: *const _ = &x.zst ;

    unsafe { rustc_peek(x) };

    let rmut_cell = unsafe {
        // Take advantage of the fact that `zst` and `cell` are at the same location in memory.
        // This trick would work with any size type if miri implemented `ptr::offset`.
        let p_cell = p_zst as *const UnsafeCell<i32>;

        let pmut_cell = (*p_cell).get();
        &mut *pmut_cell
    };

    *rmut_cell = 42;
    let val = *rmut_cell;
    unsafe { rustc_peek(x) };

    val
};

fn main() {
    println!("{}", BOO);
}

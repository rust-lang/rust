#![feature(core_intrinsics, rustc_attrs, const_raw_ptr_deref)]

use std::cell::Cell;
use std::intrinsics::rustc_peek;

#[rustc_mir(rustc_peek_indirectly_mutable, stop_after_dataflow)]
pub fn mut_ref(flag: bool) -> i32 {
    let mut i = 0;
    let cell = Cell::new(0);

    if flag {
        let p = &mut i;
        unsafe { rustc_peek(i) };
        *p = 1;

        let p = &cell;
        unsafe { rustc_peek(cell) };
        p.set(2);
    } else {
        unsafe { rustc_peek(i) };    //~ ERROR rustc_peek: bit not set
        unsafe { rustc_peek(cell) }; //~ ERROR rustc_peek: bit not set

        let p = &mut cell;
        unsafe { rustc_peek(cell) };
        *p = Cell::new(3);
    }

    unsafe { rustc_peek(i) };
    unsafe { rustc_peek(cell) };
    i + cell.get()
}

fn main() {}

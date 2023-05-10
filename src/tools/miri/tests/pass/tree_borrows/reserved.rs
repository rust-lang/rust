//@compile-flags: -Zmiri-tree-borrows -Zmiri-tag-gc=0

#[path = "../../utils/mod.rs"]
mod utils;
use utils::macros::*;
use utils::miri_extern::miri_write_to_stderr;

use std::cell::UnsafeCell;

// We exhaustively check that Reserved behaves as we want under all of the
// following conditions:
// - with or without interior mutability
// - with or without a protector
// - for a foreign read or write
// Of these cases, those in this file are the ones that must not cause
// immediate UB, and those that do are in tests/fail/tree-borrows/reserved/
// and are the combinations [_ + protected + write]

fn main() {
    unsafe {
        cell_protected_read();
        cell_unprotected_read();
        cell_unprotected_write();
        int_protected_read();
        int_unprotected_read();
        int_unprotected_write();
    }
}

unsafe fn print(msg: &str) {
    miri_write_to_stderr(msg.as_bytes());
    miri_write_to_stderr("\n".as_bytes());
}

unsafe fn read_second<T>(x: &mut T, y: *mut u8) {
    name!(x as *mut T as *mut u8=>1, "caller:x");
    name!(x as *mut T as *mut u8, "callee:x");
    name!(y, "caller:y");
    name!(y, "callee:y");
    let _val = *y;
}

// Foreign Read on a interior mutable Protected Reserved turns it Frozen.
unsafe fn cell_protected_read() {
    print("[interior mut + protected] Foreign Read: Re* -> Frz");
    let base = &mut UnsafeCell::new(0u8);
    name!(base.get(), "base");
    let alloc_id = alloc_id!(base.get());
    let x = &mut *(base as *mut UnsafeCell<u8>);
    name!(x.get(), "x");
    let y = (&mut *base).get();
    name!(y);
    read_second(x, y); // Foreign Read for callee:x
    print_state!(alloc_id);
}

// Foreign Read on an interior mutable pointer is a noop.
unsafe fn cell_unprotected_read() {
    print("[interior mut] Foreign Read: Re* -> Re*");
    let base = &mut UnsafeCell::new(0u64);
    name!(base.get(), "base");
    let alloc_id = alloc_id!(base.get());
    let x = &mut *(base as *mut UnsafeCell<_>);
    name!(x.get(), "x");
    let y = (&mut *base).get();
    name!(y);
    let _val = *y; // Foreign Read for x
    print_state!(alloc_id);
}

// Foreign Write on an interior mutable pointer is a noop.
// Also y must become Active.
unsafe fn cell_unprotected_write() {
    print("[interior mut] Foreign Write: Re* -> Re*");
    let base = &mut UnsafeCell::new(0u64);
    name!(base.get(), "base");
    let alloc_id = alloc_id!(base.get());
    let x = &mut *(base as *mut UnsafeCell<u64>);
    name!(x.get(), "x");
    let y = (&mut *base).get();
    name!(y);
    *y = 1; // Foreign Write for x
    print_state!(alloc_id);
}

// Foreign Read on a Protected Reserved turns it Frozen.
unsafe fn int_protected_read() {
    print("[protected] Foreign Read: Res -> Frz");
    let base = &mut 0u8;
    let alloc_id = alloc_id!(base);
    name!(base);
    let x = &mut *(base as *mut u8);
    name!(x);
    let y = (&mut *base) as *mut u8;
    name!(y);
    read_second(x, y); // Foreign Read for callee:x
    print_state!(alloc_id);
}

// Foreign Read on a Reserved is a noop.
// Also y must become Active.
unsafe fn int_unprotected_read() {
    print("[] Foreign Read: Res -> Res");
    let base = &mut 0u8;
    name!(base);
    let alloc_id = alloc_id!(base);
    let x = &mut *(base as *mut u8);
    name!(x);
    let y = (&mut *base) as *mut u8;
    name!(y);
    let _val = *y; // Foreign Read for x
    print_state!(alloc_id);
}

// Foreign Write on a Reserved turns it Disabled.
unsafe fn int_unprotected_write() {
    print("[] Foreign Write: Res -> Dis");
    let base = &mut 0u8;
    name!(base);
    let alloc_id = alloc_id!(base);
    let x = &mut *(base as *mut u8);
    name!(x);
    let y = (&mut *base) as *mut u8;
    name!(y);
    *y = 1; // Foreign Write for x
    print_state!(alloc_id);
}

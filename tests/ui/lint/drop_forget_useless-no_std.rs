//@ check-pass
// Check that the drop/forget lints name `core` paths in `no_std` crates.

#![no_std]
#![crate_type = "lib"]

#[derive(Copy, Clone)]
pub struct SomeStruct;

pub fn references(s: &SomeStruct) {
    core::mem::drop(s);
    //~^ WARN calls to `core::mem::drop` with a reference
    core::mem::forget(s);
    //~^ WARN calls to `core::mem::forget` with a reference
    drop(s);
    //~^ WARN calls to `core::mem::drop` with a reference
}

pub fn copy_types(s: SomeStruct) {
    core::mem::drop(s);
    //~^ WARN calls to `core::mem::drop` with a value that implements `Copy`
    core::mem::forget(s);
    //~^ WARN calls to `core::mem::forget` with a value that implements `Copy`
    drop(s);
    //~^ WARN calls to `core::mem::drop` with a value that implements `Copy`
}

pub unsafe fn drop_in_place(r: *mut &SomeStruct, c: *mut SomeStruct) {
    unsafe {
        core::ptr::drop_in_place(r);
        //~^ WARN calls to `core::ptr::drop_in_place` with a pointer to a reference
        core::ptr::drop_in_place(c);
        //~^ WARN calls to `core::ptr::drop_in_place` with a pointer to a value that implements `Copy`
        r.drop_in_place();
        //~^ WARN calls to `drop_in_place` with a pointer to a reference
        c.drop_in_place();
        //~^ WARN calls to `drop_in_place` with a pointer to a value that implements `Copy`
    }
}

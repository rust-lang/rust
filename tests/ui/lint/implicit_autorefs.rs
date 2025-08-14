//@ check-fail
//@ run-rustfix

#![allow(dead_code)] // For the rustfix-ed code.

use std::mem::ManuallyDrop;
use std::ops::Deref;

unsafe fn test_const(ptr: *const [u8]) {
    let _ = (*ptr)[..16];
    //~^ ERROR implicit autoref
}

struct Test {
    field: [u8],
}

unsafe fn test_field(ptr: *const Test) -> *const [u8] {
    let l = (*ptr).field.len();
    //~^ ERROR implicit autoref

    &raw const (*ptr).field[..l - 1]
    //~^ ERROR implicit autoref
}

unsafe fn test_builtin_index(a: *mut [String]) {
    _ = (*a)[0].len();
    //~^ ERROR implicit autoref

    _ = (*a)[..1][0].len();
    //~^ ERROR implicit autoref
    //~^^ ERROR implicit autoref
}

unsafe fn test_overloaded_deref_const(ptr: *const ManuallyDrop<Test>) {
    let _ = (*ptr).field;
    //~^ ERROR implicit autoref
    let _ = &raw const (*ptr).field;
    //~^ ERROR implicit autoref
}

unsafe fn test_overloaded_deref_mut(ptr: *mut ManuallyDrop<Test>) {
    let _ = (*ptr).field;
    //~^ ERROR implicit autoref
}

unsafe fn test_double_overloaded_deref_const(ptr: *const ManuallyDrop<ManuallyDrop<Test>>) {
    let _ = (*ptr).field;
    //~^ ERROR implicit autoref
}

unsafe fn test_manually_overloaded_deref() {
    struct W<T>(T);

    impl<T> Deref for W<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.0
        }
    }

    let w: W<i32> = W(5);
    let w = &raw const w;
    let _p: *const i32 = &raw const **w;
    //~^ ERROR implicit autoref
}

struct Test2 {
    // Derefs to `[u8]`.
    field: &'static [u8],
}

fn test_more_manual_deref(ptr: *const Test2) -> usize {
    unsafe { (*ptr).field.len() }
    //~^ ERROR implicit autoref
}

unsafe fn test_no_attr(ptr: *mut ManuallyDrop<u8>) {
    // Should not warn, as `ManuallyDrop::write` is not
    // annotated with `#[rustc_no_implicit_auto_ref]`
    ptr.write(ManuallyDrop::new(1));
}

unsafe fn test_vec_get(ptr: *mut Vec<u8>) {
    let _ = (*ptr).get(0);
    //~^ ERROR implicit autoref
    let _ = (*ptr).get_unchecked(0);
    //~^ ERROR implicit autoref
    let _ = (*ptr).get_mut(0);
    //~^ ERROR implicit autoref
    let _ = (*ptr).get_unchecked_mut(0);
    //~^ ERROR implicit autoref
}

unsafe fn test_string(ptr: *mut String) {
    let _ = (*ptr).len();
    //~^ ERROR implicit autoref
    let _ = (*ptr).is_empty();
    //~^ ERROR implicit autoref
}

unsafe fn slice_ptr_len_because_of_msrv<T>(slice: *const [T]) {
    let _ = (*slice)[..].len();
    //~^ ERROR implicit autoref
    //~^^ ERROR implicit autoref
}

fn main() {}

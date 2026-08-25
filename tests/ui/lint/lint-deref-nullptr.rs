// test the deref_nullptr lint

use std::ptr;

struct Struct {
    field: u8,
}

#[derive(Clone, Copy)]
struct Zst;

fn f() {
    unsafe {
        let a = 1;
        let ub = *(a as *const i32);
        let ub = *(0 as *const i32);
        //~^ ERROR dereferencing a null pointer
        let ub = *ptr::null::<i32>();
        //~^ ERROR dereferencing a null pointer
        let ub = *ptr::null_mut::<i32>();
        //~^ ERROR dereferencing a null pointer
        let ub = *(ptr::null::<i16>() as *const i32);
        //~^ ERROR dereferencing a null pointer
        let ub = *(ptr::null::<i16>() as *mut i32 as *mut usize as *const u8);
        //~^ ERROR dereferencing a null pointer
        let ub = &*ptr::null::<i32>();
        //~^ ERROR dereferencing a null pointer
        let ub = &*ptr::null_mut::<i32>();
        //~^ ERROR dereferencing a null pointer
        ptr::addr_of!(*ptr::null::<i32>());
        // ^^ OKAY
        ptr::addr_of_mut!(*ptr::null_mut::<i32>());
        // ^^ OKAY
        let offset = ptr::addr_of!((*ptr::null::<Struct>()).field);
        //~^ ERROR dereferencing a null pointer

        // Make sure the lint permits derefs of null pointers to ZSTs
        let ok: Zst = *ptr::null();
        let ok: Zst = *ptr::null_mut();
        let ok: Zst = *(0 as *const Zst);
    }
}

fn main() {}

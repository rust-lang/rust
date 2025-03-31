// check-fail
// run-rustfix
#![allow(unnecessary_transmutes)]

use std::{mem, ptr};

unsafe fn null_ptr() {
    ptr::write(
        //~^ ERROR calling this function with a null pointer is undefined behavior
        ptr::null_mut() as *mut u32,
        mem::transmute::<[u8; 4], _>([0, 0, 0, 255]),
    );

    let null_ptr = ptr::null_mut();
    ptr::write(
        //~^ ERROR calling this function with a null pointer is undefined behavior
        null_ptr as *mut u32,
        mem::transmute::<[u8; 4], _>([0, 0, 0, 255]),
    );

    let _: &[usize] = std::slice::from_raw_parts(ptr::null(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    let _: &[usize] = std::slice::from_raw_parts(ptr::null_mut(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    let _: &[usize] = std::slice::from_raw_parts(0 as *mut _, 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    let _: &[usize] = std::slice::from_raw_parts(mem::transmute(0usize), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    let _: &[usize] = std::slice::from_raw_parts_mut(ptr::null_mut(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::copy::<usize>(ptr::null(), ptr::NonNull::dangling().as_ptr(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    ptr::copy::<usize>(ptr::NonNull::dangling().as_ptr(), ptr::null_mut(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::copy_nonoverlapping::<usize>(ptr::null(), ptr::NonNull::dangling().as_ptr(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    ptr::copy_nonoverlapping::<usize>(
        //~^ ERROR calling this function with a null pointer is undefined behavior
        ptr::NonNull::dangling().as_ptr(),
        ptr::null_mut(),
        0,
    );

    #[derive(Copy, Clone)]
    struct A(usize);
    let mut v = A(200);

    let _a: A = ptr::read(ptr::null());
    //~^ ERROR calling this function with a null pointer is undefined behavior
    let _a: A = ptr::read(ptr::null_mut());
    //~^ ERROR calling this function with a null pointer is undefined behavior

    let _a: A = ptr::read_unaligned(ptr::null());
    //~^ ERROR calling this function with a null pointer is undefined behavior
    let _a: A = ptr::read_unaligned(ptr::null_mut());
    //~^ ERROR calling this function with a null pointer is undefined behavior

    let _a: A = ptr::read_volatile(ptr::null());
    //~^ ERROR calling this function with a null pointer is undefined behavior
    let _a: A = ptr::read_volatile(ptr::null_mut());
    //~^ ERROR calling this function with a null pointer is undefined behavior

    let _a: A = ptr::replace(ptr::null_mut(), v);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::swap::<A>(ptr::null_mut(), &mut v);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    ptr::swap::<A>(&mut v, ptr::null_mut());
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::swap_nonoverlapping::<A>(ptr::null_mut(), &mut v, 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    ptr::swap_nonoverlapping::<A>(&mut v, ptr::null_mut(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::write(ptr::null_mut(), v);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::write_unaligned(ptr::null_mut(), v);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::write_volatile(ptr::null_mut(), v);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::write_bytes::<usize>(ptr::null_mut(), 42, 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    // with indirections
    let const_ptr = null_ptr as *const u8;
    let _a: u8 = ptr::read(const_ptr);
    //~^ ERROR calling this function with a null pointer is undefined behavior
}

unsafe fn zst() {
    struct Zst; // zero-sized type

    std::slice::from_raw_parts::<()>(ptr::null(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    std::slice::from_raw_parts::<Zst>(ptr::null(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    std::slice::from_raw_parts_mut::<()>(ptr::null_mut(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior
    std::slice::from_raw_parts_mut::<Zst>(ptr::null_mut(), 0);
    //~^ ERROR calling this function with a null pointer is undefined behavior

    ptr::read::<()>(ptr::null());
    ptr::read::<Zst>(ptr::null());

    ptr::write(ptr::null_mut(), ());
    ptr::write(ptr::null_mut(), Zst);

    ptr::copy(ptr::null::<()>(), ptr::null_mut::<()>(), 1);
    ptr::copy(ptr::null::<Zst>(), ptr::null_mut::<Zst>(), 1);
}

unsafe fn not_invalid() {
    // Simplified false-positive from std quicksort implementation

    let mut a = ptr::null_mut();
    let mut b = ();

    loop {
        if false {
            break;
        }

        a = &raw mut b;
    }

    ptr::write(a, ());
}

fn main() {}

#![warn(clippy::swap_ptr_to_ref)]

use core::ptr::addr_of_mut;

fn main() {
    let mut x = 0u32;
    let y: *mut _ = &mut x;
    let z: *mut _ = &mut x;

    unsafe {
        core::mem::swap(&mut *y, &mut *z);
        //~^ swap_ptr_to_ref
        core::mem::swap(&mut *y, &mut x);
        //~^ swap_ptr_to_ref
        core::mem::swap(&mut x, &mut *y);
        //~^ swap_ptr_to_ref
        core::mem::swap(&mut *addr_of_mut!(x), &mut *addr_of_mut!(x));
        //~^ swap_ptr_to_ref
    }

    let y = &mut x;
    let mut z = 0u32;
    let z = &mut z;

    core::mem::swap(y, z);
}

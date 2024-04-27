#![warn(clippy::swap_ptr_to_ref)]

macro_rules! addr_of_mut_to_ref {
    ($e:expr) => {
        &mut *core::ptr::addr_of_mut!($e)
    };
}

fn main() {
    let mut x = 0u32;
    let y: *mut _ = &mut x;

    unsafe {
        core::mem::swap(addr_of_mut_to_ref!(x), &mut *y);
        //~^ ERROR: call to `core::mem::swap` with a parameter derived from a raw pointer
        //~| NOTE: `-D clippy::swap-ptr-to-ref` implied by `-D warnings`
        core::mem::swap(&mut *y, addr_of_mut_to_ref!(x));
        //~^ ERROR: call to `core::mem::swap` with a parameter derived from a raw pointer
        core::mem::swap(addr_of_mut_to_ref!(x), addr_of_mut_to_ref!(x));
        //~^ ERROR: call to `core::mem::swap` with a parameter derived from a raw pointer
    }
}

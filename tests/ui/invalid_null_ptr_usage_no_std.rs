#![no_std]
#![feature(lang_items)]

use core::panic::PanicInfo;

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    loop {}
}

fn main() {
    unsafe {
        let _slice: &[usize] = core::slice::from_raw_parts(core::ptr::null(), 0);
        //~^ invalid_null_ptr_usage
        let _slice: &[usize] = core::slice::from_raw_parts(core::ptr::null_mut(), 0);
        //~^ invalid_null_ptr_usage

        let _slice: &[usize] = core::slice::from_raw_parts_mut(core::ptr::null_mut(), 0);
        //~^ invalid_null_ptr_usage

        core::ptr::copy::<usize>(core::ptr::null(), core::ptr::NonNull::dangling().as_ptr(), 0);
        //~^ invalid_null_ptr_usage
        core::ptr::copy::<usize>(core::ptr::NonNull::dangling().as_ptr(), core::ptr::null_mut(), 0);
        //~^ invalid_null_ptr_usage

        core::ptr::copy_nonoverlapping::<usize>(core::ptr::null(), core::ptr::NonNull::dangling().as_ptr(), 0);
        //~^ invalid_null_ptr_usage
        core::ptr::copy_nonoverlapping::<usize>(core::ptr::NonNull::dangling().as_ptr(), core::ptr::null_mut(), 0);
        //~^ invalid_null_ptr_usage

        struct A; // zero sized struct
        assert_eq!(core::mem::size_of::<A>(), 0);

        let _a: A = core::ptr::read(core::ptr::null());
        //~^ invalid_null_ptr_usage
        let _a: A = core::ptr::read(core::ptr::null_mut());
        //~^ invalid_null_ptr_usage

        let _a: A = core::ptr::read_unaligned(core::ptr::null());
        //~^ invalid_null_ptr_usage
        let _a: A = core::ptr::read_unaligned(core::ptr::null_mut());
        //~^ invalid_null_ptr_usage

        let _a: A = core::ptr::read_volatile(core::ptr::null());
        //~^ invalid_null_ptr_usage
        let _a: A = core::ptr::read_volatile(core::ptr::null_mut());
        //~^ invalid_null_ptr_usage

        let _a: A = core::ptr::replace(core::ptr::null_mut(), A);
        //~^ invalid_null_ptr_usage
        let _slice: *const [usize] = core::ptr::slice_from_raw_parts(core::ptr::null_mut(), 0); // shouldn't lint
        let _slice: *const [usize] = core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut(), 0);

        core::ptr::swap::<A>(core::ptr::null_mut(), &mut A);
        //~^ invalid_null_ptr_usage
        core::ptr::swap::<A>(&mut A, core::ptr::null_mut());
        //~^ invalid_null_ptr_usage

        core::ptr::swap_nonoverlapping::<A>(core::ptr::null_mut(), &mut A, 0);
        //~^ invalid_null_ptr_usage
        core::ptr::swap_nonoverlapping::<A>(&mut A, core::ptr::null_mut(), 0);
        //~^ invalid_null_ptr_usage

        core::ptr::write(core::ptr::null_mut(), A);
        //~^ invalid_null_ptr_usage

        core::ptr::write_unaligned(core::ptr::null_mut(), A);
        //~^ invalid_null_ptr_usage

        core::ptr::write_volatile(core::ptr::null_mut(), A);
        //~^ invalid_null_ptr_usage

        core::ptr::write_bytes::<usize>(core::ptr::null_mut(), 42, 0);
        //~^ invalid_null_ptr_usage
    }
}

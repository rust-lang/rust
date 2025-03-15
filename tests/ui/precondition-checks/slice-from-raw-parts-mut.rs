//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: slice::from_raw_parts_mut requires
//@ revisions: null misaligned toolarge

#![allow(invalid_null_arguments)]

fn main() {
    unsafe {
        #[cfg(null)]
        let _s: &mut [u8] = std::slice::from_raw_parts_mut(std::ptr::null_mut(), 0);
        #[cfg(misaligned)]
        let _s: &mut [u16] = std::slice::from_raw_parts_mut(1usize as *mut u16, 0);
        #[cfg(toolarge)]
        let _s: &mut [u16] =
            std::slice::from_raw_parts_mut(2usize as *mut u16, isize::MAX as usize);
    }
}

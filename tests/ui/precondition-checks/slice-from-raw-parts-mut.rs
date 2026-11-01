//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ revisions: null misaligned toolarge
//@ [misaligned] error-pattern: unsafe precondition(s) violated: slice::from_raw_parts_mut requires
//@ [toolarge] error-pattern: unsafe precondition(s) violated: slice::from_raw_parts_mut requires
//@ [misaligned] run-crash
//@ [toolarge] run-crash
//@ [null] run-pass

#![allow(invalid_null_arguments)]

fn main() {
    unsafe {
        // This is okay
        #[cfg(null)]
        let _s: &mut [u8] = std::slice::from_raw_parts_mut(std::ptr::null_mut(), 0);
        #[cfg(misaligned)]
        let _s: &mut [u16] = std::slice::from_raw_parts_mut(1usize as *mut u16, 0);
        #[cfg(toolarge)]
        let _s: &mut [u16] =
            std::slice::from_raw_parts_mut(2usize as *mut u16, isize::MAX as usize);
    }
}

//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ revisions: null misaligned toolarge
//@ [misaligned] error-pattern: unsafe precondition(s) violated: slice::from_raw_parts requires
//@ [toolarge] error-pattern: unsafe precondition(s) violated: slice::from_raw_parts requires
//@ [misaligned] run-crash
//@ [toolarge] run-crash
//@ [null] run-pass

#![allow(invalid_null_arguments)]

fn main() {
    unsafe {
        // This is okay.
        #[cfg(null)]
        let _s: &[u8] = std::slice::from_raw_parts(std::ptr::null(), 0);
        #[cfg(misaligned)]
        let _s: &[u16] = std::slice::from_raw_parts(1usize as *const u16, 0);
        #[cfg(toolarge)]
        let _s: &[u16] = std::slice::from_raw_parts(2usize as *const u16, isize::MAX as usize);
    }
}

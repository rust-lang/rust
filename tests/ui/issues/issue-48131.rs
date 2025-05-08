// This note is annotated because the purpose of the test
// is to ensure that certain other notes are not generated.
#![deny(unused_unsafe)]


// (test that no note is generated on this unsafe fn)
pub unsafe fn a() {
    fn inner() {
        unsafe { /* unnecessary */ } //~ ERROR unnecessary `unsafe`
    }

    inner()
}

pub fn b() {
    // (test that no note is generated on this unsafe block)
    unsafe {
        fn inner() {
            unsafe { /* unnecessary */ } //~ ERROR unnecessary `unsafe`
        }
        // `()` is fine to zero-initialize as it is zero sized and inhabited.
        let () = ::std::mem::zeroed();

        inner()
    }
}

fn main() {}

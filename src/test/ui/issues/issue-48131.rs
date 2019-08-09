// This note is annotated because the purpose of the test
// is to ensure that certain other notes are not generated.
#![deny(unused_unsafe)] //~ NOTE
#![allow(deprecated)]

// (test that no note is generated on this unsafe fn)
pub unsafe fn a() {
    fn inner() {
        unsafe { /* unnecessary */ } //~ ERROR unnecessary `unsafe`
                                     //~^ NOTE
    }

    inner()
}

pub fn b() {
    // (test that no note is generated on this unsafe block)
    unsafe {
        fn inner() {
            unsafe { /* unnecessary */ } //~ ERROR unnecessary `unsafe`
                                         //~^ NOTE
        }

        let () = ::std::mem::uninitialized();

        inner()
    }
}

fn main() {}

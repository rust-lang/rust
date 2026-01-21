// Regression test for <https://github.com/rust-lang/rust/issues/149542>.
//
// This checks that a nested type ascription doesn't cause a crash when the
// compiler checks if it constitutes a read of the never type.
//
//@ check-pass

#![feature(never_type)]
#![feature(type_ascription)]
#![deny(unreachable_code)]

fn main() {
    unsafe {
        let _ = type_ascribe!(type_ascribe!(*std::ptr::null(), !), _);

        // this is *not* unreachable, because previous line does not actually read the never type
        ();
    }
}

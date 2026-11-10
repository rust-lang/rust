//@ run-pass
//@ needs-unwind
#![allow(dead_code)]
//@ compile-flags: -C codegen-units=3
//@ needs-threads

// Test unwinding through multiple compilation units.

// According to acrichto, in the distant past `ld -r` (which is used during
// linking when codegen-units > 1) was known to produce object files with
// damaged unwinding tables.  This may be related to GNU binutils bug #6893
// ("Partial linking results in corrupt .eh_frame_hdr"), but I'm not certain.
// In any case, this test should let us know if enabling parallel codegen ever
// breaks unwinding.


use std::thread;

fn pad() -> usize { 0 }

mod a {
    pub fn f() {
        panic!();
    }
}

mod b {
    pub fn g() {
        crate::a::f();
    }
}

fn main() {
    thread::spawn(move|| { b::g() }).join().unwrap_err();
}

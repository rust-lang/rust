// compile-flags: -O
// only-x86_64

#![crate_type = "rlib"]

use std::arch::asm;

// CHECK-LABEL: @x87_clobber
// CHECK: ~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)}
#[no_mangle]
pub unsafe fn x87_clobber() {
    asm!("foo", out("st") _);
}

// CHECK-LABEL: @mmx_clobber
// CHECK: ~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)}
#[no_mangle]
pub unsafe fn mmx_clobber() {
    asm!("bar", out("mm0") _, out("mm1") _);
}

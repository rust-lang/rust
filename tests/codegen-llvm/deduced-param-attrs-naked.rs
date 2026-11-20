// The motivating example for naked_functions_rustic_abi (jump to a concrete impl) doesn't work if
// parameter deduction is allowed to modify the ABI of the naked function. This test checks that we
// don't deduce parameter attributes for that case.

//@ compile-flags: -O -Cno-prepopulate-passes
//@ only-x86_64

#![crate_type = "lib"]
#![feature(naked_functions_rustic_abi)]

pub struct Big {
    pub blah: [i32; 1024],
}

// CHECK-LABEL: @external(ptr {{.*}}readonly {{.*}}captures(none) {{.*}}%_b)
#[unsafe(no_mangle)]
pub fn external(_b: Big) {}

// CHECK:       declare void @naked(
// CHECK-NOT:   readonly
// CHECK-NOT:   captures(none)
// CHECK-SAME:  ) unnamed_addr
#[unsafe(no_mangle)]
#[unsafe(naked)]
pub fn naked(_b: Big) {
    core::arch::naked_asm!("jmp {}", sym impl_of_naked)
}

#[unsafe(no_mangle)]
pub fn impl_of_naked(mut b: Big) {
    std::hint::black_box(b);
}

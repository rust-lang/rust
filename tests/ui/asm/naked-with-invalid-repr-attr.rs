//@ needs-asm-support

// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

#![crate_type = "lib"]
use std::arch::naked_asm;

#[repr(C)]
//~^ ERROR attribute cannot be used on
#[unsafe(naked)]
extern "C" fn example1() {
    naked_asm!("")
}

#[repr(transparent)]
//~^ ERROR attribute cannot be used on
#[unsafe(naked)]
extern "C" fn example2() {
    naked_asm!("")
}

#[repr(C)]
//~^ ERROR attribute cannot be used on
#[rustc_align(16)]
#[unsafe(naked)]
extern "C" fn example3() {
    naked_asm!("")
}

// note: two errors because of packed and C
#[repr(C, packed)]
//~^ ERROR attribute cannot be used on
//~| ERROR attribute cannot be used on
//~| NOTE duplicate diagnostic emitted due to
#[unsafe(naked)]
extern "C" fn example4() {
    naked_asm!("")
}

#[repr(u8)]
//~^ ERROR `#[repr]` attribute cannot be used on
#[unsafe(naked)]
extern "C" fn example5() {
    naked_asm!("")
}

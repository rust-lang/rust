//@ needs-asm-support
#![feature(naked_functions)]
#![feature(fn_align)]
#![crate_type = "lib"]
use std::arch::asm;

#[repr(C)]
//~^ ERROR attribute should be applied to a struct, enum, or union [E0517]
#[naked]
extern "C" fn example1() {
    //~^ NOTE not a struct, enum, or union
    unsafe { asm!("", options(noreturn)) }
}

#[repr(transparent)]
//~^ ERROR attribute should be applied to a struct, enum, or union [E0517]
#[naked]
extern "C" fn example2() {
    //~^ NOTE not a struct, enum, or union
    unsafe { asm!("", options(noreturn)) }
}

#[repr(align(16), C)]
//~^ ERROR attribute should be applied to a struct, enum, or union [E0517]
#[naked]
extern "C" fn example3() {
    //~^ NOTE not a struct, enum, or union
    unsafe { asm!("", options(noreturn)) }
}

// note: two errors because of packed and C
#[repr(C, packed)]
//~^ ERROR attribute should be applied to a struct or union [E0517]
//~| ERROR attribute should be applied to a struct, enum, or union [E0517]
#[naked]
extern "C" fn example4() {
    //~^ NOTE not a struct, enum, or union
    //~| NOTE not a struct or union
    unsafe { asm!("", options(noreturn)) }
}

#[repr(u8)]
//~^ ERROR attribute should be applied to an enum [E0517]
#[naked]
extern "C" fn example5() {
    //~^ NOTE not an enum
    unsafe { asm!("", options(noreturn)) }
}

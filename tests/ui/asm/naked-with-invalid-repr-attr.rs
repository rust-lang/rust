//@ needs-asm-support

// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

#![crate_type = "lib"]
use std::arch::naked_asm;

#[repr(C)]
//~^ ERROR attribute should be applied to a struct, enum, or union [E0517]
#[unsafe(naked)]
extern "C" fn example1() {
    //~^ NOTE not a struct, enum, or union
    naked_asm!("")
}

#[repr(transparent)]
//~^ ERROR attribute should be applied to a struct, enum, or union [E0517]
#[unsafe(naked)]
extern "C" fn example2() {
    //~^ NOTE not a struct, enum, or union
    naked_asm!("")
}

#[repr(C)]
//~^ ERROR attribute should be applied to a struct, enum, or union [E0517]
#[rustc_align(16)]
#[unsafe(naked)]
extern "C" fn example3() {
    //~^ NOTE not a struct, enum, or union
    naked_asm!("")
}

// note: two errors because of packed and C
#[repr(C, packed)]
//~^ ERROR attribute should be applied to a struct or union [E0517]
//~| ERROR attribute should be applied to a struct, enum, or union [E0517]
#[unsafe(naked)]
extern "C" fn example4() {
    //~^ NOTE not a struct, enum, or union
    //~| NOTE not a struct or union
    naked_asm!("")
}

#[repr(u8)]
//~^ ERROR attribute should be applied to an enum [E0517]
#[unsafe(naked)]
extern "C" fn example5() {
    //~^ NOTE not an enum
    naked_asm!("")
}

// check-pass
// only-x86_64
#![feature(asm)]
#![feature(naked_functions)]
#![crate_type = "lib"]

#[naked]
pub extern "C" fn naked(p: char) -> u128 {
    //~^ WARN uses type `char`
    //~| WARN uses type `u128`
    unsafe { asm!("", options(noreturn)); }
}

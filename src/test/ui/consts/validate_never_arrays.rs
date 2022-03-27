// stderr-per-bitwidth
#![feature(never_type)]

const _: &[!; 1] = unsafe { &*(1_usize as *const [!; 1]) };
//~^ ERROR evaluation of constant value failed
const _: &[!; 0] = unsafe { &*(1_usize as *const [!; 0]) }; // ok
const _: &[!] = unsafe { &*(1_usize as *const [!; 0]) }; // ok
const _: &[!] = unsafe { &*(1_usize as *const [!; 1]) };
//~^ ERROR evaluation of constant value failed
const _: &[!] = unsafe { &*(1_usize as *const [!; 42]) };
//~^ ERROR evaluation of constant value failed

fn main() {}

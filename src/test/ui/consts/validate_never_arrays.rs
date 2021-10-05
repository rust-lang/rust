// stderr-per-bitwidth
#![feature(never_type)]

const _: &[!; 1] = unsafe { &*(1_usize as *const [!; 1]) }; //~ ERROR undefined behavior
const _: &[!; 0] = unsafe { &*(1_usize as *const [!; 0]) }; // ok
const _: &[!] = unsafe { &*(1_usize as *const [!; 0]) }; // ok
const _: &[!] = unsafe { &*(1_usize as *const [!; 1]) }; //~ ERROR undefined behavior
const _: &[!] = unsafe { &*(1_usize as *const [!; 42]) }; //~ ERROR undefined behavior

fn main() {}

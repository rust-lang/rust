#![feature(global_asm)]

#[cfg(target_arch = "x86")]
global_asm!("");

#[cfg(target_arch = "x86_64")]
global_asm!("");

#[cfg(target_arch = "arm")]
global_asm!("");

#[cfg(target_arch = "aarch64")]
global_asm!("");

#[cfg(target_arch = "mips")]
global_asm!("");

fn main() {}

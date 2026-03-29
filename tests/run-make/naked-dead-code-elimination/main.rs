#![feature(cfg_target_object_format)]
use std::arch::naked_asm;

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "C" fn used() {
    naked_asm!("ret")
}

#[unsafe(no_mangle)]
extern "C" fn unused_clothed() -> i32 {
    42
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "C" fn unused() {
    naked_asm!("ret")
}

#[unsafe(naked)]
#[cfg_attr(target_object_format = "macho", unsafe(link_section = "__TEXT,foobar"))]
#[cfg_attr(not(target_object_format = "macho"), unsafe(link_section = ".foobar"))]
#[unsafe(no_mangle)]
extern "C" fn unused_link_section() {
    naked_asm!("ret")
}

#[cfg_attr(target_object_format = "macho", unsafe(link_section = "__TEXT,baz"))]
#[cfg_attr(not(target_object_format = "macho"), unsafe(link_section = ".baz"))]
#[unsafe(no_mangle)]
extern "C" fn unused_link_section_clothed() -> i32 {
    43
}

fn main() {
    used();
}

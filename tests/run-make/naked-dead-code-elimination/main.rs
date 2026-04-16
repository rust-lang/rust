#![feature(cfg_target_object_format)]
use std::arch::naked_asm;

#[unsafe(naked)]
#[unsafe(no_mangle)]
extern "C" fn used() {
    naked_asm!("ret")
}

#[unsafe(no_mangle)]
extern "C" fn used_clothed() -> i32 {
    41
}

pub fn main() {
    std::hint::black_box(used());
    std::hint::black_box(used_clothed());
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
#[unsafe(link_section = cfg_select!(
    target_object_format = "mach-o" => "__TEXT,foobar",
    _ => ".foobar",
))]
#[unsafe(no_mangle)]
extern "C" fn unused_link_section() {
    naked_asm!("ret")
}

#[unsafe(link_section = cfg_select!(
    target_object_format = "mach-o" => "__TEXT,baz",
    _ => ".baz",
))]
#[unsafe(no_mangle)]
extern "C" fn unused_link_section_clothed() -> i32 {
    43
}

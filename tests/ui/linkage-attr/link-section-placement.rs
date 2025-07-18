//! Test placement of functions and statics in custom link sections

//@ run-pass

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]
#![allow(non_upper_case_globals)]
#[cfg(not(target_vendor = "apple"))]
#[link_section = ".moretext"]
fn i_live_in_more_text() -> &'static str {
    "knock knock"
}

#[cfg(not(target_vendor = "apple"))]
#[link_section = ".imm"]
static magic: usize = 42;

#[cfg(not(target_vendor = "apple"))]
#[link_section = ".mut"]
static mut frobulator: usize = 0xdeadbeef;

#[cfg(target_vendor = "apple")]
#[link_section = "__TEXT,__moretext"]
fn i_live_in_more_text() -> &'static str {
    "knock knock"
}

#[cfg(target_vendor = "apple")]
#[link_section = "__RODATA,__imm"]
static magic: usize = 42;

#[cfg(target_vendor = "apple")]
#[link_section = "__DATA,__mut"]
static mut frobulator: usize = 0xdeadbeef;

pub fn main() {
    unsafe {
        frobulator = 0x12345678;
        println!("{} {} {}", i_live_in_more_text(), magic, frobulator);
    }
}

#![allow(non_upper_case_globals)]
#[cfg(not(target_os = "macos"))]
#[link_section=".moretext"]
fn i_live_in_more_text() -> &'static str {
    "knock knock"
}

#[cfg(not(target_os = "macos"))]
#[link_section=".imm"]
static magic: usize = 42;

#[cfg(not(target_os = "macos"))]
#[link_section=".mut"]
static mut frobulator: usize = 0xdeadbeef;

#[cfg(target_os = "macos")]
#[link_section="__TEXT,__moretext"]
fn i_live_in_more_text() -> &'static str {
    "knock knock"
}

#[cfg(target_os = "macos")]
#[link_section="__RODATA,__imm"]
static magic: usize = 42;

#[cfg(target_os = "macos")]
#[link_section="__DATA,__mut"]
static mut frobulator: usize = 0xdeadbeef;

pub fn main() {
    unsafe {
        frobulator = 0xcafebabe;
        println!("{} {} {}", i_live_in_more_text(), magic, frobulator);
    }
}

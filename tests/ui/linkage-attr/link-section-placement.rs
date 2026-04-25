//! Test placement of functions and statics in custom link sections

//@ run-pass

#![feature(cfg_target_object_format)]
// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]
#![allow(non_upper_case_globals)]

cfg_select! {
    target_object_format = "mach-o" => {
        #[link_section = "__TEXT,__moretext"]
        fn i_live_in_more_text() -> &'static str {
            "knock knock"
        }

        #[link_section = "__RODATA,__imm"]
        static magic: usize = 42;

        #[link_section = "__DATA,__mut"]
        static mut frobulator: usize = 0xdeadbeef;
    }
    _ => {
        #[link_section = ".moretext"]
        fn i_live_in_more_text() -> &'static str {
            "knock knock"
        }

        #[link_section = ".imm"]
        static magic: usize = 42;

        #[link_section = ".mut"]
        static mut frobulator: usize = 0xdeadbeef;
    }
}

pub fn main() {
    unsafe {
        frobulator = 0x12345678;
        println!("{} {} {}", i_live_in_more_text(), magic, frobulator);
    }
}

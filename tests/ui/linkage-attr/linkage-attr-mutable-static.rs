//! The symbols are resolved by the linker. It doesn't make sense to change
//! them at runtime, so deny mutable statics with #[linkage].

#![feature(linkage)]

fn main() {
    extern "C" {
        #[linkage = "weak"] //~ ERROR mutable statics are not allowed with `#[linkage]`
        static mut ABC: *const u8;
    }

    unsafe {
        assert_eq!(ABC as usize, 0);
    }
}

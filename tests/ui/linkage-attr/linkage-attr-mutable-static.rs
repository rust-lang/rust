//! The symbols are resolved by the linker. It doesn't make sense to change
//! them at runtime, so deny mutable statics with #[linkage].

#![feature(linkage)]

fn main() {
    #[rustfmt::skip]
    extern "C" {
        #[linkage = "extern_weak"] //~ ERROR extern mutable statics are not allowed with `#[linkage]`
        static mut EXTERN_WEAK: *const u8;
    }

    unsafe {
        assert_eq!(EXTERN_WEAK as usize, 0);
    }

    // static mut is fine here as this is a definition rather than declaration.
    #[linkage = "weak"]
    static mut WEAK_DEF: u8 = 42;

    unsafe {
        assert_eq!(WEAK_DEF, 0);
    }
}

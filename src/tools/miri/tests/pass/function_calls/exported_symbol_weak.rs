#![feature(linkage)]

// FIXME move this module to a separate crate once aux-build is allowed
// This currently depends on the fact that miri skips the codegen check
// that denies multiple symbols with the same name.
mod first {
    #[no_mangle]
    #[linkage = "weak"]
    extern "C" fn foo() -> i32 {
        1
    }

    #[no_mangle]
    #[linkage = "weak"]
    extern "C" fn bar() -> i32 {
        2
    }
}

mod second {
    #[no_mangle]
    extern "C" fn bar() -> i32 {
        3
    }
}

extern "C" {
    fn foo() -> i32;
    fn bar() -> i32;
}

fn main() {
    unsafe {
        // If there is no non-weak definition, the weak definition will be used.
        assert_eq!(foo(), 1);
        // Non-weak definition takes presedence.
        assert_eq!(bar(), 3);
    }
}

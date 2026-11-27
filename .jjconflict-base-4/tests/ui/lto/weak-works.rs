//@ run-pass

//@ compile-flags: -C codegen-units=8 -Z thinlto
//@ ignore-i686-pc-windows-gnu
//@ ignore-x86_64-pc-windows-gnu

#![feature(linkage)]

pub mod foo {
    #[linkage = "weak"]
    #[no_mangle]
    pub extern "C" fn FOO() -> i32 {
        0
    }
}

mod bar {
    extern "C" {
        fn FOO() -> i32;
    }

    pub fn bar() -> i32 {
        unsafe { FOO() }
    }
}

fn main() {
    bar::bar();
}

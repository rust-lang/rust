#![feature(export_stable)]
#![crate_type = "sdylib"]

pub mod m {
    #[repr(C)]
    pub struct S {
        pub x: i32,
    }

    pub extern "C" fn foo1(x: S) -> i32 {
        x.x
    }
}

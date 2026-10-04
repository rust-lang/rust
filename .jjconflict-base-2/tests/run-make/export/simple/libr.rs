#![feature(export_stable)]
#![crate_type = "sdylib"]

#[export_stable]
pub mod m {
    #[repr(C)]
    pub struct S {
        pub x: i32,
    }

    pub extern "C" fn foo1(x: S) -> i32 {
        x.x
    }

    pub type Integer = i32;

    impl S {
        pub extern "C" fn foo2(x: Integer) -> Integer {
            x
        }
    }
}

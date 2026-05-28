#![feature(extern_types)]

pub mod aaaaaaa {

    extern "C" {
        pub type MyForeignType;
    }

    impl MyForeignType {
        pub fn my_method() {}
    }
}

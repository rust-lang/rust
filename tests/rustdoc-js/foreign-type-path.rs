#![feature(extern_types)]

pub mod aaaaaaa {

    extern {
        pub type MyForeignType;
    }

    impl MyForeignType {
        pub fn my_method() {}
    }

}

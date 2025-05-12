#![feature(decl_macro)]

mod inner1 {
    pub struct Struct {}

    pub mod inner2 {
        pub macro mac() {
            super::Struct
        }
    }
}

pub use inner1::inner2 as public;

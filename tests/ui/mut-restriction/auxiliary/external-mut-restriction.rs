#![feature(mut_restriction)]

pub struct TopLevel {
    pub mut(crate) x: i32,
}

pub mod inner {
    pub struct Inner {
        pub mut(self) x: i32,
    }
}

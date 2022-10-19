#![feature(mut_restriction)]

pub struct TopLevel {
    pub mut(self) alpha: u8,
}

impl TopLevel {
    pub fn new() -> Self {
        Self { alpha: 0 }
    }
}

pub mod inner {
    pub struct Inner {
        pub mut(self) beta: u8,
    }

    impl Inner {
        pub fn new() -> Self {
            Self { beta: 0 }
        }
    }
}

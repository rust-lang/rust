#![feature(mut_restriction)]

#[derive(Default)]
pub struct TopLevelStruct {
    pub mut(self) field: u8,
}

#[derive(Default)]
pub enum TopLevelEnum {
    #[default]
    Default,
    A(mut(self) u8),
    B { mut(self) field: u8 },
}

pub mod inner {
    #[derive(Default)]
    pub struct InnerStruct {
        pub mut(self) field: u8,
    }

    #[derive(Default)]
    pub enum InnerEnum {
        #[default]
        Default,
        A(mut(self) u8),
        B { mut(self) field: u8 },
    }
}

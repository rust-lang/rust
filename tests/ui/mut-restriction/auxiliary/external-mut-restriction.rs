#![feature(mut_restriction)]

#[derive(Default)]
pub struct TopLevelS {
    pub mut(crate) x: i32,
    pub y: i32,
}

pub enum TopLevelE {
    Var {
        mut(crate) x: i32,
        y: i32,
    },
    Tup(mut(crate) i32, i32),
}

pub union TopLevelU {
    pub mut(crate) x: i32,
    pub y: i32,
}

pub mod inner {
    #[derive(Default)]
    pub struct InnerS {
        pub mut(self) x: i32,
        pub y: i32,
    }

    pub enum InnerE {
        Var {
            mut(self) x: i32,
            y: i32,
        },
        Tup(mut(self) i32, i32),
    }

    pub union InnerU {
        pub mut(self) x: i32,
        pub y: i32,
    }
}

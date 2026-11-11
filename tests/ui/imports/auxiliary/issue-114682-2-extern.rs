macro_rules! m {
    () => {
        pub fn max() {}
        pub(crate) mod max {}
    };
}

mod d {
    m! {}
}

mod e {
    pub type max = i32;
}

pub use self::d::*;
pub use self::e::*;

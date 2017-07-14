pub mod a {
    pub struct Abc;
}

pub mod b {
    pub use a::*;
}

pub mod c {
    pub use a::Abc;
}

pub use self::a::Abc;

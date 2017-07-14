pub mod a {
    pub struct Abc;
}

pub mod b {
    pub use a::*;
}

pub mod a {
    pub use super::b::Abc;
}

pub mod b {
    pub struct Abc;
}

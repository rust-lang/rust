mod a {
    pub type Result<T> = std::result::Result<T, ()>;
}

mod b {
    pub type Result<T> = std::result::Result<T, ()>;
}

pub use a::*;
pub use b::*;

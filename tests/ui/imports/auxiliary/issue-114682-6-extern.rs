mod a {
    pub fn log() {}
}
mod b {
    pub fn log() {}
}

pub use self::a::*;
pub use self::b::*;

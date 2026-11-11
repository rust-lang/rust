#![crate_name = "inner"]

pub mod ser {
    pub trait Serialize {}
    pub trait Serializer {}
}

pub mod de {
    pub trait Deserialize {}
}

pub use ser::{Serialize, Serializer};
pub use de::*;

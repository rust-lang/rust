pub struct S;
mod m { pub struct S; }
pub use crate::m::S as S2;

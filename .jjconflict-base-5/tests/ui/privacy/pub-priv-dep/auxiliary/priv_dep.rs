pub struct OtherType;
pub trait OtherTrait {}
impl OtherTrait for OtherType {}

#[macro_export]
macro_rules! m {
    () => {};
}

pub enum E {
    V1
}

struct PrivType;

pub type Unit = ();
pub type PubPub = OtherType;
pub type PubPriv = PrivType;

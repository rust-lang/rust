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

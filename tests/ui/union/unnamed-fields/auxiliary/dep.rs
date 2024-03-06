#[repr(C)]
pub struct GoodStruct(());

pub struct BadStruct(());

pub enum BadEnum {
    A,
    B,
}

#[repr(C)]
pub enum BadEnum2 {
    A,
    B,
}

pub type GoodAlias = GoodStruct;
pub type BadAlias = i32;

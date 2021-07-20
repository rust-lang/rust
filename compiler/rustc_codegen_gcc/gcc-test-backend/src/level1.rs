use mini_core::Sync;

pub struct Struct3 {
    pub field1: &'static [u8],
    pub field2: i32,
}

unsafe impl Sync for Struct3 {}

pub static STRUCT3: Struct3 = Struct3 {
    field1: b"level1",
    field2: 1,
};

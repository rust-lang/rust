pub enum HiddenEnum {
    A,
    B,
    #[doc(hidden)]
    C,
}

#[derive(Default)]
pub struct HiddenStruct {
    pub one: u8,
    pub two: bool,
    #[doc(hidden)]
    pub hide: usize,
}

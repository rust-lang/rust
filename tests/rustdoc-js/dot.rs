pub struct Bar {
    pub start: u32,
    pub end: u32,
}

impl Bar {
    pub const START: u32 = 12;
    pub const END: u32 = 12;

    pub fn start(&self) {}
    pub fn end() {}
}

pub trait Another {
    const START: u32;
    type End;

    fn start(&self) {}
    fn end();
}

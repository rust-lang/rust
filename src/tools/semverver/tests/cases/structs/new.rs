#![allow(dead_code)]
pub struct Abc<B> {
    field: B,
}

pub struct Def<A=u8> {
    pub field: A,
}

pub struct Def2<A=u16> {
    pub field: A,
}

pub struct Efg {
    pub field: u16,
}

pub struct Fgh {
    pub field: u8,
}

pub struct Ghi {
    field: u8,
}

pub struct Hij {
    field: u8,
}

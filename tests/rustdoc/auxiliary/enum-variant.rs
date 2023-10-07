#![crate_name = "bar"]

pub enum E {
    A = 12,
    B,
    C = 1245,
}

pub enum F {
    A,
    B,
}

#[repr(u32)]
pub enum G {
    A = 12,
    B,
    C(u32),
}

pub enum H {
    A,
    C(u32),
}

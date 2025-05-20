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

#[repr(C)]
pub enum N {
    A,
    B,
}

#[repr(C)]
pub enum O {
    A(u32),
    B,
}

#[repr(u32)]
pub enum P {
    A,
    B,
}

#[repr(u32)]
pub enum Q {
    A(u32),
    B,
}

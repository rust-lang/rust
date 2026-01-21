#![crate_name = "diamond_base"]
#![crate_type = "rlib"]

#[cfg(rpass1)]
fn private_impl() -> u32 {
    42
}

#[cfg(any(rpass2, rpass3))]
fn private_impl() -> u32 {
    21 + 21
}

#[cfg(rpass3)]
fn _another_private() -> u32 {
    999
}

pub fn base_value() -> u32 {
    private_impl()
}

pub struct BaseStruct {
    pub value: u32,
}

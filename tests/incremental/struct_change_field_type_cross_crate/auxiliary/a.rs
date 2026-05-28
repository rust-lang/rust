#![crate_type="rlib"]

 #[cfg(rpass1)]
pub struct X {
    pub x: u32
}

#[cfg(rpass2)]
pub struct X {
    pub x: i32
}

pub struct EmbedX {
    pub x: X
}

pub struct Y {
    pub y: char
}

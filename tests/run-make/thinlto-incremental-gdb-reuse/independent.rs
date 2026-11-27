#![no_std]

pub mod a {
    pub fn value() -> u32 {
        if cfg!(change_a) { 3 } else { 1 }
    }
}

pub mod b {
    pub fn value() -> u32 {
        if cfg!(change_b) { 4 } else { 2 }
    }
}

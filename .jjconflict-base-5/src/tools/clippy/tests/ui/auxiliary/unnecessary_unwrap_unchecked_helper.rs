#![allow(unused, clippy::missing_safety_doc)]

pub fn lol() -> Option<u32> {
    Some(0)
}

pub unsafe fn lol_unchecked() -> u32 {
    0
}

pub fn kek() -> Option<u32> {
    Some(0)
}

unsafe fn kek_unchecked() -> u32 {
    0
}

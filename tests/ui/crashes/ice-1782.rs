#![allow(dead_code, unused_variables)]

/// Should not trigger an ICE in `SpanlessEq` / `consts::constant`
///
/// Issue: https://github.com/rust-lang/rust-clippy/issues/1782
use std::{mem, ptr};

fn spanless_eq_ice() {
    let txt = "something";
    match txt {
        "something" => unsafe {
            ptr::write(
                ptr::null_mut() as *mut u32,
                mem::transmute::<[u8; 4], _>([0, 0, 0, 255]),
            )
        },
        _ => unsafe {
            ptr::write(
                ptr::null_mut() as *mut u32,
                mem::transmute::<[u8; 4], _>([13, 246, 24, 255]),
            )
        },
    }
}

fn main() {}

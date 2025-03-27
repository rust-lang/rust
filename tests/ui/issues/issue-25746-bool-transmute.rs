//@ run-pass
#![allow(unnecessary_transmutes)]
use std::mem::transmute;

fn main() {
    unsafe {
        let _: i8 = transmute(false);
        let _: i8 = transmute(true);
        let _: bool = transmute(0u8);
        let _: bool = transmute(1u8);
    }
}

//@ run-rustfix
#![deny(unnecessary_transmutes)]
#![allow(unused_unsafe, unused_imports, unused_variables, unused_parens)]
use std::mem::transmute;

pub fn bytes_at_home(x: u32) -> [u8; 4] {
    unsafe { transmute(x) }
    //~^ ERROR
}

fn main() {
    unsafe {
        let x: u16 = transmute(*b"01");
        //~^ ERROR
        let x: [u8; 2] = transmute(x);
        //~^ ERROR
        let x: u32 = transmute(*b"0123");
        //~^ ERROR
        let x: [u8; 4] = transmute(x);
        //~^ ERROR
        let x: u64 = transmute(*b"feriscat");
        //~^ ERROR
        let x: [u8; 8] = transmute(x);
        //~^ ERROR

        let y: i16 = transmute(*b"01");
        //~^ ERROR
        let y: [u8; 2] = transmute(y);
        //~^ ERROR
        let y: i32 = transmute(*b"0123");
        //~^ ERROR
        let y: [u8; 4] = transmute(y);
        //~^ ERROR
        let y: i64 = transmute(*b"feriscat");
        //~^ ERROR
        let y: [u8; 8] = transmute(y);
        //~^ ERROR

        let z: f32 = transmute(*b"0123");
        //~^ ERROR
        let z: [u8; 4] = transmute(z);
        //~^ ERROR
        let z: f64 = transmute(*b"feriscat");
        //~^ ERROR
        let z: [u8; 8] = transmute(z);
        //~^ ERROR

        let y: u32 = transmute('🦀');
        //~^ ERROR
        let y: char = transmute(y);
        //~^ ERROR

        let x: u16 = transmute(8i16);
        //~^ ERROR
        let x: i16 = transmute(x);
        //~^ ERROR
        let x: u32 = transmute(4i32);
        //~^ ERROR
        let x: i32 = transmute(x);
        //~^ ERROR
        let x: u64 = transmute(7i64);
        //~^ ERROR
        let x: i64 = transmute(x);
        //~^ ERROR

        let y: f32 = transmute(1u32);
        //~^ ERROR
        let y: u32 = transmute(y);
        //~^ ERROR
        let y: f64 = transmute(3u64);
        //~^ ERROR
        let y: u64 = transmute(2.0);
        //~^ ERROR

        let z: bool = transmute(1u8);
        //~^ ERROR
        let z: u8 = transmute(z);
        //~^ ERROR

        let z: bool = transmute(1i8);
        // no error!
        let z: i8 = transmute(z);
        //~^ ERROR
    }
}

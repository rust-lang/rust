//@ aux-build:proc_macros.rs

#![warn(clippy::host_endian_bytes)]
#![feature(f16, f128)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

#[inline_macros]
fn main() {
    {
        let _ = 0u8.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0u16.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0u32.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0u64.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0u128.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0usize.to_ne_bytes(); //~ host_endian_bytes

        let _ = 0i8.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0i16.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0i32.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0i64.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0i128.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0isize.to_ne_bytes(); //~ host_endian_bytes

        let _ = 0f16.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0f32.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0f64.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0f128.to_ne_bytes(); //~ host_endian_bytes

        let _ = u8::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = u16::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = u32::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = u64::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = u128::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = usize::from_ne_bytes([0; _]); //~ host_endian_bytes

        let _ = i8::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = i16::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = i32::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = i64::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = i128::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = isize::from_ne_bytes([0; _]); //~ host_endian_bytes

        let _ = f16::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = f32::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = f64::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = f128::from_ne_bytes([0; _]); //~ host_endian_bytes
    };

    {
        let _ = u8::from_ne_bytes; //~ host_endian_bytes
        let _ = u8::to_ne_bytes; //~ host_endian_bytes
    }

    #[warn(clippy::big_endian_bytes)]
    {
        let _ = 0u8.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0i8.to_ne_bytes(); //~ host_endian_bytes
        let _ = u8::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = i8::from_ne_bytes([0; _]); //~ host_endian_bytes
    };

    #[warn(clippy::little_endian_bytes)]
    {
        let _ = 0u8.to_ne_bytes(); //~ host_endian_bytes
        let _ = 0i8.to_ne_bytes(); //~ host_endian_bytes
        let _ = u8::from_ne_bytes([0; _]); //~ host_endian_bytes
        let _ = i8::from_ne_bytes([0; _]); //~ host_endian_bytes
    };

    {
        inline! {{
            let _ = 0u8.to_ne_bytes(); //~ host_endian_bytes
            let _ = u8::from_ne_bytes([0; _]); //~ host_endian_bytes
            let _ = 0u8.$to_ne_bytes(); //~ host_endian_bytes
            let _ = u8::$from_ne_bytes([0; _]); //~ host_endian_bytes
        }}

        external! {
            let _ = 0u8.to_ne_bytes();
            let _ = u8::from_ne_bytes([0; _]);
            let _ = 0u8.$to_ne_bytes(); //~ host_endian_bytes
            let _ = u8::$from_ne_bytes([0; _]); //~ host_endian_bytes
        }
    }
}

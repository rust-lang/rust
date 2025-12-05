#![allow(unused)]
#![allow(clippy::diverging_sub_expression)]
#![no_main]

// FIXME(f16_f128): add these types when `{to_from}_*_bytes` are available

macro_rules! fn_body {
    () => {
        2u8.to_ne_bytes();
        //~^ host_endian_bytes
        2i8.to_ne_bytes();
        //~^ host_endian_bytes
        2u16.to_ne_bytes();
        //~^ host_endian_bytes
        2i16.to_ne_bytes();
        //~^ host_endian_bytes
        2u32.to_ne_bytes();
        //~^ host_endian_bytes
        2i32.to_ne_bytes();
        //~^ host_endian_bytes
        2u64.to_ne_bytes();
        //~^ host_endian_bytes
        2i64.to_ne_bytes();
        //~^ host_endian_bytes
        2u128.to_ne_bytes();
        //~^ host_endian_bytes
        2i128.to_ne_bytes();
        //~^ host_endian_bytes
        2.0f32.to_ne_bytes();
        //~^ host_endian_bytes
        2.0f64.to_ne_bytes();
        //~^ host_endian_bytes
        2usize.to_ne_bytes();
        //~^ host_endian_bytes
        2isize.to_ne_bytes();
        //~^ host_endian_bytes
        u8::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        i8::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        u16::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        i16::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        u32::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        i32::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        u64::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        i64::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        u128::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        i128::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        usize::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        isize::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        f32::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        f64::from_ne_bytes(todo!());
        //~^ host_endian_bytes

        2u8.to_le_bytes();
        //~^ little_endian_bytes
        2i8.to_le_bytes();
        //~^ little_endian_bytes
        2u16.to_le_bytes();
        //~^ little_endian_bytes
        2i16.to_le_bytes();
        //~^ little_endian_bytes
        2u32.to_le_bytes();
        //~^ little_endian_bytes
        2i32.to_le_bytes();
        //~^ little_endian_bytes
        2u64.to_le_bytes();
        //~^ little_endian_bytes
        2i64.to_le_bytes();
        //~^ little_endian_bytes
        2u128.to_le_bytes();
        //~^ little_endian_bytes
        2i128.to_le_bytes();
        //~^ little_endian_bytes
        2.0f32.to_le_bytes();
        //~^ little_endian_bytes
        2.0f64.to_le_bytes();
        //~^ little_endian_bytes
        2usize.to_le_bytes();
        //~^ little_endian_bytes
        2isize.to_le_bytes();
        //~^ little_endian_bytes
        u8::from_le_bytes(todo!());
        //~^ little_endian_bytes
        i8::from_le_bytes(todo!());
        //~^ little_endian_bytes
        u16::from_le_bytes(todo!());
        //~^ little_endian_bytes
        i16::from_le_bytes(todo!());
        //~^ little_endian_bytes
        u32::from_le_bytes(todo!());
        //~^ little_endian_bytes
        i32::from_le_bytes(todo!());
        //~^ little_endian_bytes
        u64::from_le_bytes(todo!());
        //~^ little_endian_bytes
        i64::from_le_bytes(todo!());
        //~^ little_endian_bytes
        u128::from_le_bytes(todo!());
        //~^ little_endian_bytes
        i128::from_le_bytes(todo!());
        //~^ little_endian_bytes
        usize::from_le_bytes(todo!());
        //~^ little_endian_bytes
        isize::from_le_bytes(todo!());
        //~^ little_endian_bytes
        f32::from_le_bytes(todo!());
        //~^ little_endian_bytes
        f64::from_le_bytes(todo!());
        //~^ little_endian_bytes
    };
}

// bless breaks if I use fn_body too much (oops)
macro_rules! fn_body_smol {
    () => {
        2u8.to_ne_bytes();
        //~^ host_endian_bytes
        //~| host_endian_bytes
        //~| host_endian_bytes
        //~| host_endian_bytes
        //~| host_endian_bytes
        u8::from_ne_bytes(todo!());
        //~^ host_endian_bytes
        //~| host_endian_bytes
        //~| host_endian_bytes
        //~| host_endian_bytes
        //~| host_endian_bytes

        2u8.to_le_bytes();
        //~^ little_endian_bytes
        //~| little_endian_bytes
        //~| little_endian_bytes
        //~| little_endian_bytes
        //~| little_endian_bytes
        u8::from_le_bytes(todo!());
        //~^ little_endian_bytes
        //~| little_endian_bytes
        //~| little_endian_bytes
        //~| little_endian_bytes
        //~| little_endian_bytes

        2u8.to_be_bytes();
        //~^ big_endian_bytes
        //~| big_endian_bytes
        //~| big_endian_bytes
        //~| big_endian_bytes
        //~| big_endian_bytes
        u8::from_be_bytes(todo!());
        //~^ big_endian_bytes
        //~| big_endian_bytes
        //~| big_endian_bytes
        //~| big_endian_bytes
        //~| big_endian_bytes
    };
}

#[rustfmt::skip]
#[warn(clippy::host_endian_bytes)]
fn host() { fn_body!(); }

#[rustfmt::skip]
#[warn(clippy::little_endian_bytes)]
fn little() { fn_body!(); }

#[rustfmt::skip]
#[warn(clippy::big_endian_bytes)]
fn big() { fn_body!(); }

#[rustfmt::skip]
#[warn(clippy::host_endian_bytes)]
#[warn(clippy::big_endian_bytes)]
fn host_encourage_little() { fn_body_smol!(); }

#[rustfmt::skip]
#[warn(clippy::host_endian_bytes)]
#[warn(clippy::little_endian_bytes)]
fn host_encourage_big() { fn_body_smol!(); }

#[rustfmt::skip]
#[warn(clippy::host_endian_bytes)]
#[warn(clippy::little_endian_bytes)]
#[warn(clippy::big_endian_bytes)]
fn no_help() { fn_body_smol!(); }

#[rustfmt::skip]
#[warn(clippy::little_endian_bytes)]
#[warn(clippy::big_endian_bytes)]
fn little_encourage_host() { fn_body_smol!(); }

#[rustfmt::skip]
#[warn(clippy::host_endian_bytes)]
#[warn(clippy::little_endian_bytes)]
fn little_encourage_big() { fn_body_smol!(); }

#[rustfmt::skip]
#[warn(clippy::big_endian_bytes)]
#[warn(clippy::little_endian_bytes)]
fn big_encourage_host() { fn_body_smol!(); }

#[rustfmt::skip]
#[warn(clippy::host_endian_bytes)]
#[warn(clippy::big_endian_bytes)]
fn big_encourage_little() { fn_body_smol!(); }

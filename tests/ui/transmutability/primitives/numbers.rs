#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]
#![allow(incomplete_features)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;
    struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context>
    {}
}

fn should_accept_identity() {
    assert::is_transmutable::<   i8,    i8>();
    assert::is_transmutable::<   u8,    u8>();
    assert::is_transmutable::<  i16,   i16>();
    assert::is_transmutable::<  u16,   u16>();
    assert::is_transmutable::<  i32,   i32>();
    assert::is_transmutable::<  f32,   f32>();
    assert::is_transmutable::<  u32,   u32>();
    assert::is_transmutable::<  i64,   i64>();
    assert::is_transmutable::<  f64,   f64>();
    assert::is_transmutable::<  u64,   u64>();
    assert::is_transmutable::< i128,  i128>();
    assert::is_transmutable::< u128,  u128>();
    assert::is_transmutable::<isize, isize>();
    assert::is_transmutable::<usize, usize>();
}

fn should_be_bitransmutable() {
    assert::is_transmutable::<   i8,    u8>();
    assert::is_transmutable::<   u8,    i8>();

    assert::is_transmutable::<  i16,   u16>();
    assert::is_transmutable::<  u16,   i16>();

    assert::is_transmutable::<  i32,   f32>();
    assert::is_transmutable::<  i32,   u32>();
    assert::is_transmutable::<  f32,   i32>();
    assert::is_transmutable::<  f32,   u32>();
    assert::is_transmutable::<  u32,   i32>();
    assert::is_transmutable::<  u32,   f32>();

    assert::is_transmutable::<  u64,   i64>();
    assert::is_transmutable::<  u64,   f64>();
    assert::is_transmutable::<  i64,   u64>();
    assert::is_transmutable::<  i64,   f64>();
    assert::is_transmutable::<  f64,   u64>();
    assert::is_transmutable::<  f64,   i64>();

    assert::is_transmutable::< u128,  i128>();
    assert::is_transmutable::< i128,  u128>();

    assert::is_transmutable::<isize, usize>();
    assert::is_transmutable::<usize, isize>();
}

fn should_reject_extension() {
    assert::is_transmutable::<   i8,   i16>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,   u16>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,   i32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,   f32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,   u32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,   u64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,   i64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,   f64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   i8,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<   u8,   i16>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,   u16>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,   i32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,   f32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,   u32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,   u64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,   i64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,   f64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<   u8,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  i16,   i32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i16,   f32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i16,   u32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i16,   u64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i16,   i64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i16,   f64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i16,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i16,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  u16,   i32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u16,   f32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u16,   u32>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u16,   u64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u16,   i64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u16,   f64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u16,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u16,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  i32,   u64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i32,   i64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i32,   f64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i32,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i32,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  f32,   u64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  f32,   i64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  f32,   f64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  f32,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  f32,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  u32,   u64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u32,   i64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u32,   f64>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u32,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u32,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  u64,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  u64,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  i64,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  i64,  i128>(); //~ ERROR cannot be safely transmuted

    assert::is_transmutable::<  f64,  u128>(); //~ ERROR cannot be safely transmuted
    assert::is_transmutable::<  f64,  i128>(); //~ ERROR cannot be safely transmuted
}

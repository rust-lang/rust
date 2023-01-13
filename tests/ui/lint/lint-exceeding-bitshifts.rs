// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O
// build-pass
// ignore-pass (test emits codegen-time warnings and verifies that they are not errors)
// normalize-stderr-test "shift left by `(64|32)_usize`, which" -> "shift left by `%BITS%`, which"

#![crate_type="lib"]
#![warn(arithmetic_overflow)]


pub trait Foo {
    const N: i32;
}

impl<T: Foo> Foo for Vec<T> {
    const N: i32 = T::N << 42; //~ WARN: arithmetic operation will overflow
}

pub fn foo(x: i32) {
    let _ = x << 42; //~ WARN: arithmetic operation will overflow
}

pub fn main() {
      let n = 1u8 << 7;
      let n = 1u8 << 8;   //~ WARN: arithmetic operation will overflow
      let n = 1u16 << 15;
      let n = 1u16 << 16; //~ WARN: arithmetic operation will overflow
      let n = 1u32 << 31;
      let n = 1u32 << 32; //~ WARN: arithmetic operation will overflow
      let n = 1u64 << 63;
      let n = 1u64 << 64; //~ WARN: arithmetic operation will overflow
      let n = 1i8 << 7;
      let n = 1i8 << 8;   //~ WARN: arithmetic operation will overflow
      let n = 1i16 << 15;
      let n = 1i16 << 16; //~ WARN: arithmetic operation will overflow
      let n = 1i32 << 31;
      let n = 1i32 << 32; //~ WARN: arithmetic operation will overflow
      let n = 1i64 << 63;
      let n = 1i64 << 64; //~ WARN: arithmetic operation will overflow

      let n = 1u8 >> 7;
      let n = 1u8 >> 8;   //~ WARN: arithmetic operation will overflow
      let n = 1u16 >> 15;
      let n = 1u16 >> 16; //~ WARN: arithmetic operation will overflow
      let n = 1u32 >> 31;
      let n = 1u32 >> 32; //~ WARN: arithmetic operation will overflow
      let n = 1u64 >> 63;
      let n = 1u64 >> 64; //~ WARN: arithmetic operation will overflow
      let n = 1i8 >> 7;
      let n = 1i8 >> 8;   //~ WARN: arithmetic operation will overflow
      let n = 1i16 >> 15;
      let n = 1i16 >> 16; //~ WARN: arithmetic operation will overflow
      let n = 1i32 >> 31;
      let n = 1i32 >> 32; //~ WARN: arithmetic operation will overflow
      let n = 1i64 >> 63;
      let n = 1i64 >> 64; //~ WARN: arithmetic operation will overflow

      let n = 1u8;
      let n = n << 7;
      let n = n << 8; //~ WARN: arithmetic operation will overflow

      let n = 1u8 << -8; //~ WARN: arithmetic operation will overflow

      let n = 1i8<<(1isize+-1);

      let n = 1u8 << (4+3);
      let n = 1u8 << (4+4); //~ WARN: arithmetic operation will overflow
      let n = 1i64 >> [63][0];
      let n = 1i64 >> [64][0]; //~ WARN: arithmetic operation will overflow

      #[cfg(target_pointer_width = "32")]
      const BITS: usize = 32;
      #[cfg(target_pointer_width = "64")]
      const BITS: usize = 64;
      let n = 1_isize << BITS; //~ WARN: arithmetic operation will overflow
      let n = 1_usize << BITS; //~ WARN: arithmetic operation will overflow
}

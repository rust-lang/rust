#![warn(clippy::mismatched_bit_width_type)]

use core::num::{self, NonZero, NonZeroI32, NonZeroU32};

fn main() {
    // left and right ewpression have different calling types
    // unsigned
    let w: u8 = 5;
    let _ = i8::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let _ = u32::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let w: u16 = 5;
    let _ = i16::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let _ = u8::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let w: u32 = 5;
    let _ = i32::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let _ = isize::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let w: u64 = 5;
    let _ = i64::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<u64>::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let w: usize = 5;
    let _ = isize::BITS - w.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<isize>::BITS - w.leading_zeros(); //~ mismatched_bit_width_type

    // signed
    let x: i8 = -5;
    let _ = u8::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let _ = i16::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let x: i16 = -5;
    let _ = u16::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<u32>::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let x: i32 = -5;
    let _ = u32::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<i32>::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let x: i64 = -5;
    let _ = u64::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let _ = isize::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let x: isize = -5;
    let _ = usize::BITS - x.leading_zeros(); //~ mismatched_bit_width_type
    let _ = u32::BITS - x.leading_zeros(); //~ mismatched_bit_width_type

    // `NonZero::<T>::BITS - x.leading_zeros()`
    // unsigned
    let y = NonZero::<u8>::new(5).unwrap();
    let _ = NonZero::<i8>::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let _ = u8::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let y = NonZero::<u16>::new(5).unwrap();
    let _ = NonZero::<i16>::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let _ = i16::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let y = NonZero::<u32>::new(5).unwrap();
    let _ = NonZeroI32::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<u64>::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let y = NonZero::<u64>::new(5).unwrap();
    let _ = NonZero::<i64>::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<usize>::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let y = NonZero::<usize>::new(5).unwrap();
    let _ = num::NonZero::<isize>::BITS - y.leading_zeros(); //~ mismatched_bit_width_type
    let _ = u64::BITS - y.leading_zeros(); //~ mismatched_bit_width_type

    // signed
    let z = NonZero::<i8>::new(-5).unwrap();
    let _ = NonZero::<u8>::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let _ = u8::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let z = NonZero::<i16>::new(-5).unwrap();
    let _ = NonZero::<u16>::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let _ = i16::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let z = NonZero::<i32>::new(-5).unwrap();
    let _ = NonZeroU32::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<u32>::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let z = NonZero::<i64>::new(-5).unwrap();
    let _ = NonZero::<u64>::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let _ = NonZero::<i32>::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let z = NonZero::<isize>::new(-5).unwrap();
    let _ = num::NonZero::<usize>::BITS - z.leading_zeros(); //~ mismatched_bit_width_type
    let _ = num::NonZero::<i64>::BITS - z.leading_zeros(); //~ mismatched_bit_width_type

    // negative case.
    // left expression is a literal
    let z: u32 = 1_000_000 - x.leading_zeros();
}

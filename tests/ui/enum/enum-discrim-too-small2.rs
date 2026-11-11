#![deny(overflowing_literals)]
#![allow(dead_code)]

#[repr(i8)]
enum Ei8 {
    Ai8 = 23,
    Bi8 = -23,
    Ci8 = 223, //~ ERROR literal out of range for `i8`
}

#[repr(i16)]
enum Ei16 {
    Ai16 = 23,
    Bi16 = -22333,
    Ci16 = 55555, //~ ERROR literal out of range for `i16`
}

#[repr(i32)]
enum Ei32 {
    Ai32 = 23,
    Bi32 = -2_000_000_000,
    Ci32 = 3_000_000_000, //~ ERROR literal out of range for `i32`
}

#[repr(i64)]
enum Ei64 {
    Ai64 = 23,
    Bi64 = -9223372036854775808,
    Ci64 = 9223372036854775809, //~ ERROR literal out of range for `i64`
}

// u64 currently allows negative numbers, and i64 allows numbers greater than `1<<63`.  This is a
// little counterintuitive, but since the discriminant can store all the bits, and extracting it
// with a cast requires specifying the signedness, there is no loss of information in those cases.
// This also applies to isize and usize on 64-bit targets.

pub fn main() { }

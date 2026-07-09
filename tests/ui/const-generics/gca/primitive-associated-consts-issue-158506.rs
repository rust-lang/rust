//@ check-pass

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

struct TakesU8<const N: u8>;
struct TakesI8<const N: i8>;

type _U8Max = TakesU8<{ u8::MAX }>;
type _U8Min = TakesU8<{ u8::MIN }>;
type _I8Max = TakesI8<{ i8::MAX }>;
type _I8Min = TakesI8<{ i8::MIN }>;

fn main() {}

// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// asmjs can't even pass i128 as arguments or return values, so ignore it.
// this will hopefully be fixed by the LLVM 5 upgrade (#43370)
// ignore-asmjs
// ignore-emscripten

// compile-flags: -Z lower_128bit_ops -C debug_assertions=no

#![feature(i128_type)]

fn test_signed(mut x: i128) -> i128 {
    x += 1;
    x -= 2;
    x *= 3;
    x /= 4;
    x %= 5;
    x <<= 6;
    x >>= 7;
    x
}

fn test_unsigned(mut x: u128) -> u128 {
    x += 1;
    x -= 2;
    x *= 3;
    x /= 4;
    x %= 5;
    x <<= 6;
    x >>= 7;
    x
}

fn main() {
    assert_eq!(test_signed(-222), -1);
    assert_eq!(test_unsigned(200), 2);
}

// END RUST SOURCE

// START rustc.test_signed.Lower128Bit.after.mir
//     _1 = const compiler_builtins::int::addsub::rust_i128_add(_1, const 1i128) -> bb7;
//     ...
//     _1 = const compiler_builtins::int::sdiv::rust_i128_div(_1, const 4i128) -> bb8;
//     ...
//     _1 = const compiler_builtins::int::sdiv::rust_i128_rem(_1, const 5i128) -> bb11;
//     ...
//     _1 = const compiler_builtins::int::mul::rust_i128_mul(_1, const 3i128) -> bb5;
//     ...
//     _1 = const compiler_builtins::int::addsub::rust_i128_sub(_1, const 2i128) -> bb6;
//     ...
//     _11 = const 7i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_i128_shr(_1, move _11) -> bb9;
//     ...
//     _12 = const 6i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_i128_shl(_1, move _12) -> bb10;
// END rustc.test_signed.Lower128Bit.after.mir

// START rustc.test_unsigned.Lower128Bit.after.mir
//     _1 = const compiler_builtins::int::addsub::rust_u128_add(_1, const 1u128) -> bb5;
//     ...
//     _1 = const compiler_builtins::int::udiv::rust_u128_div(_1, const 4u128) -> bb6;
//     ...
//     _1 = const compiler_builtins::int::udiv::rust_u128_rem(_1, const 5u128) -> bb9;
//     ...
//     _1 = const compiler_builtins::int::mul::rust_u128_mul(_1, const 3u128) -> bb3;
//     ...
//     _1 = const compiler_builtins::int::addsub::rust_u128_sub(_1, const 2u128) -> bb4;
//     ...
//     _5 = const 7i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_u128_shr(_1, move _5) -> bb7;
//     ...
//     _6 = const 6i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_u128_shl(_1, move _6) -> bb8;
// END rustc.test_unsigned.Lower128Bit.after.mir

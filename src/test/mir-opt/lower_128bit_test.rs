// ignore-emscripten

// compile-flags: -Z lower_128bit_ops=yes -C debug_assertions=no -O

static TEST_SIGNED: i128 = const_signed(-222);
static TEST_UNSIGNED: u128 = const_unsigned(200);

const fn const_signed(mut x: i128) -> i128 {
    ((((((x + 1) - 2) * 3) / 4) % 5) << 6) >> 7
}

const fn const_unsigned(mut x: u128) -> u128 {
    ((((((x + 1) - 2) * 3) / 4) % 5) << 6) >> 7
}

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

fn check(x: i128, y: u128) {
    assert_eq!(test_signed(x), -1);
    assert_eq!(const_signed(x), -1);
    assert_eq!(TEST_SIGNED, -1);
    assert_eq!(test_unsigned(y), 2);
    assert_eq!(const_unsigned(y), 2);
    assert_eq!(TEST_UNSIGNED, 2);
}

fn main() {
    check(-222, 200);
}

// END RUST SOURCE

// START rustc.const_signed.Lower128Bit.after.mir
// _7 = const compiler_builtins::int::addsub::rust_i128_add(move _8, const 1i128) -> bb7;
// ...
// _10 = Eq(const 4i128, const -1i128);
// _11 = Eq(_5, const -170141183460469231731687303715884105728i128);
// _12 = BitAnd(move _10, move _11);
// assert(!move _12, "attempt to divide with overflow") -> bb2;
// ...
// _4 = const compiler_builtins::int::sdiv::rust_i128_div(move _5, const 4i128) -> bb8;
// ...
// _14 = Eq(const 5i128, const -1i128);
// _15 = Eq(_4, const -170141183460469231731687303715884105728i128);
// _16 = BitAnd(move _14, move _15);
// assert(!move _16, "attempt to calculate the remainder with overflow") -> bb4;
// ...
// _3 = const compiler_builtins::int::sdiv::rust_i128_rem(move _4, const 5i128) -> bb11;
// ...
// _9 = Eq(const 4i128, const 0i128);
// assert(!move _9, "attempt to divide by zero") -> bb1;
// ...
// _5 = const compiler_builtins::int::mul::rust_i128_mul(move _6, const 3i128) -> bb5;
// ...
// _6 = const compiler_builtins::int::addsub::rust_i128_sub(move _7, const 2i128) -> bb6;
// ...
// _13 = Eq(const 5i128, const 0i128);
// assert(!move _13, "attempt to calculate the remainder with a divisor of zero") -> bb3;
// ...
// _17 = const 7i32 as u32 (Misc);
// _0 = const compiler_builtins::int::shift::rust_i128_shr(move _2, move _17) -> bb9;
// ...
// _18 = const 6i32 as u32 (Misc);
// _2 = const compiler_builtins::int::shift::rust_i128_shl(move _3, move _18) -> bb10;
// END rustc.const_signed.Lower128Bit.after.mir

// START rustc.const_unsigned.Lower128Bit.after.mir
// _8 = _1;
// _7 = const compiler_builtins::int::addsub::rust_u128_add(move _8, const 1u128) -> bb5;
// ...
// _4 = const compiler_builtins::int::udiv::rust_u128_div(move _5, const 4u128) -> bb6;
// ...
// _3 = const compiler_builtins::int::udiv::rust_u128_rem(move _4, const 5u128) -> bb9;
// ...
// _9 = Eq(const 4u128, const 0u128);
// assert(!move _9, "attempt to divide by zero") -> bb1;
// ...
// _5 = const compiler_builtins::int::mul::rust_u128_mul(move _6, const 3u128) -> bb3;
// ...
// _6 = const compiler_builtins::int::addsub::rust_u128_sub(move _7, const 2u128) -> bb4;
// ...
// _10 = Eq(const 5u128, const 0u128);
// assert(!move _10, "attempt to calculate the remainder with a divisor of zero") -> bb2;
// ...
// return;
// ...
// _11 = const 7i32 as u32 (Misc);
// _0 = const compiler_builtins::int::shift::rust_u128_shr(move _2, move _11) -> bb7;
// ...
// _12 = const 6i32 as u32 (Misc);
// _2 = const compiler_builtins::int::shift::rust_u128_shl(move _3, move _12) -> bb8;

// END rustc.const_unsigned.Lower128Bit.after.mir

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
//     _10 = const 7i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_i128_shr(_1, move _10) -> bb9;
//     ...
//     _11 = const 6i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_i128_shl(_1, move _11) -> bb10;
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
//     _4 = const 7i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_u128_shr(_1, move _4) -> bb7;
//     ...
//     _5 = const 6i32 as u32 (Misc);
//     _1 = const compiler_builtins::int::shift::rust_u128_shl(_1, move _5) -> bb8;
// END rustc.test_unsigned.Lower128Bit.after.mir

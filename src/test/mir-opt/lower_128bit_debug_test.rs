// asmjs can't even pass i128 as arguments or return values, so ignore it.
// this will hopefully be fixed by the LLVM 5 upgrade (#43370)
// ignore-asmjs
// ignore-emscripten

// compile-flags: -Z lower_128bit_ops=yes -C debug_assertions=yes

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
//     _8 = _1;
//     _9 = const compiler_builtins::int::addsub::rust_i128_addo(move _8, const 1i128) -> bb10;
//     ...
//     _7 = move (_9.0: i128);
//     ...
//     _10 = const compiler_builtins::int::addsub::rust_i128_subo(move _7, const 2i128) -> bb11;
//     ...
//     _6 = move (_10.0: i128);
//     ...
//     _11 = const compiler_builtins::int::mul::rust_i128_mulo(move _6, const 3i128) -> bb12;
//     ...
//     _5 = move (_11.0: i128);
//     ...
//     _12 = Eq(const 4i128, const 0i128);
//     assert(!move _12, "attempt to divide by zero") -> bb4;
//     ...
//     _13 = Eq(const 4i128, const -1i128);
//     _14 = Eq(_5, const -170141183460469231731687303715884105728i128);
//     _15 = BitAnd(move _13, move _14);
//     assert(!move _15, "attempt to divide with overflow") -> bb5;
//     ...
//     _4 = const compiler_builtins::int::sdiv::rust_i128_div(move _5, const 4i128) -> bb13;
//     ...
//     _17 = Eq(const 5i128, const -1i128);
//     _18 = Eq(_4, const -170141183460469231731687303715884105728i128);
//     _19 = BitAnd(move _17, move _18);
//     assert(!move _19, "attempt to calculate the remainder with overflow") -> bb7;
//     ...
//     _3 = const compiler_builtins::int::sdiv::rust_i128_rem(move _4, const 5i128) -> bb15;
//     ...
//     _2 = move (_20.0: i128);
//     ...
//     _23 = const 7i32 as u128 (Misc);
//     _21 = const compiler_builtins::int::shift::rust_i128_shro(move _2, move _23) -> bb16;
//     ...
//     _0 = move (_21.0: i128);
//     ...
//     assert(!move (_9.1: bool), "attempt to add with overflow") -> bb1;
//     ...
//     assert(!move (_10.1: bool), "attempt to subtract with overflow") -> bb2;
//     ...
//     assert(!move (_11.1: bool), "attempt to multiply with overflow") -> bb3;
//     ...
//     _16 = Eq(const 5i128, const 0i128);
//     assert(!move _16, "attempt to calculate the remainder with a divisor of zero") -> bb6;
//     ...
//     assert(!move (_20.1: bool), "attempt to shift left with overflow") -> bb8;
//     ...
//     _22 = const 6i32 as u128 (Misc);
//     _20 = const compiler_builtins::int::shift::rust_i128_shlo(move _3, move _22) -> bb14;
//     ...
//     assert(!move (_21.1: bool), "attempt to shift right with overflow") -> bb9;
// END rustc.const_signed.Lower128Bit.after.mir

// START rustc.const_unsigned.Lower128Bit.after.mir
//     _8 = _1;
//     _9 = const compiler_builtins::int::addsub::rust_u128_addo(move _8, const 1u128) -> bb8;
//     ...
//     _7 = move (_9.0: u128);
//     ...
//     _10 = const compiler_builtins::int::addsub::rust_u128_subo(move _7, const 2u128) -> bb9;
//     ...
//     _6 = move (_10.0: u128);
//     ...
//     _11 = const compiler_builtins::int::mul::rust_u128_mulo(move _6, const 3u128) -> bb10;
//     ...
//     _5 = move (_11.0: u128);
//     ...
//     _12 = Eq(const 4u128, const 0u128);
//     assert(!move _12, "attempt to divide by zero") -> bb4;
//     ...
//     _4 = const compiler_builtins::int::udiv::rust_u128_div(move _5, const 4u128) -> bb11;
//     ...
//     _3 = const compiler_builtins::int::udiv::rust_u128_rem(move _4, const 5u128) -> bb13;
//     ...
//     _2 = move (_14.0: u128);
//     ...
//     _17 = const 7i32 as u128 (Misc);
//     _15 = const compiler_builtins::int::shift::rust_u128_shro(move _2, move _17) -> bb14;
//     ...
//     _0 = move (_15.0: u128);
//     ...
//     assert(!move (_9.1: bool), "attempt to add with overflow") -> bb1;
//     ...
//     assert(!move (_10.1: bool), "attempt to subtract with overflow") -> bb2;
//     ...
//     assert(!move (_11.1: bool), "attempt to multiply with overflow") -> bb3;
//     ...
//     _13 = Eq(const 5u128, const 0u128);
//     assert(!move _13, "attempt to calculate the remainder with a divisor of zero") -> bb5;
//     ...
//     assert(!move (_14.1: bool), "attempt to shift left with overflow") -> bb6;
//     ...
//     _16 = const 6i32 as u128 (Misc);
//     _14 = const compiler_builtins::int::shift::rust_u128_shlo(move _3, move _16) -> bb12;
//     ...
//     assert(!move (_15.1: bool), "attempt to shift right with overflow") -> bb7;
// END rustc.const_unsigned.Lower128Bit.after.mir

// START rustc.test_signed.Lower128Bit.after.mir
//     _2 = const compiler_builtins::int::addsub::rust_i128_addo(_1, const 1i128) -> bb10;
//     ...
//     _1 = move (_2.0: i128);
//     _3 = const compiler_builtins::int::addsub::rust_i128_subo(_1, const 2i128) -> bb11;
//     ...
//     _1 = move (_3.0: i128);
//     _4 = const compiler_builtins::int::mul::rust_i128_mulo(_1, const 3i128) -> bb12;
//     ...
//     _1 = move (_4.0: i128);
//     ...
//     _1 = const compiler_builtins::int::sdiv::rust_i128_div(_1, const 4i128) -> bb13;
//     ...
//     _1 = const compiler_builtins::int::sdiv::rust_i128_rem(_1, const 5i128) -> bb15;
//     ...
//     _1 = move (_13.0: i128);
//     ...
//     _16 = const 7i32 as u128 (Misc);
//     _14 = const compiler_builtins::int::shift::rust_i128_shro(_1, move _16) -> bb16;
//     ...
//     _1 = move (_14.0: i128);
//     ...
//     assert(!move (_2.1: bool), "attempt to add with overflow") -> bb1;
//     ...
//     assert(!move (_3.1: bool), "attempt to subtract with overflow") -> bb2;
//     ...
//     assert(!move (_4.1: bool), "attempt to multiply with overflow") -> bb3;
//     ...
//     assert(!move (_13.1: bool), "attempt to shift left with overflow") -> bb8;
//     ...
//     _15 = const 6i32 as u128 (Misc);
//     _13 = const compiler_builtins::int::shift::rust_i128_shlo(_1, move _15) -> bb14;
//     ...
//     assert(!move (_14.1: bool), "attempt to shift right with overflow") -> bb9;
// END rustc.test_signed.Lower128Bit.after.mir

// START rustc.test_unsigned.Lower128Bit.after.mir
//     _2 = const compiler_builtins::int::addsub::rust_u128_addo(_1, const 1u128) -> bb8;
//     ...
//     _1 = move (_2.0: u128);
//     _3 = const compiler_builtins::int::addsub::rust_u128_subo(_1, const 2u128) -> bb9;
//     ...
//     _1 = move (_3.0: u128);
//     _4 = const compiler_builtins::int::mul::rust_u128_mulo(_1, const 3u128) -> bb10;
//     ...
//     _1 = move (_4.0: u128);
//     ...
//     _1 = const compiler_builtins::int::udiv::rust_u128_div(_1, const 4u128) -> bb11;
//     ...
//     _1 = const compiler_builtins::int::udiv::rust_u128_rem(_1, const 5u128) -> bb13;
//     ...
//     _1 = move (_7.0: u128);
//     ...
//     _10 = const 7i32 as u128 (Misc);
//     _8 = const compiler_builtins::int::shift::rust_u128_shro(_1, move _10) -> bb14;
//     ...
//     _1 = move (_8.0: u128);
//     ...
//     assert(!move (_2.1: bool), "attempt to add with overflow") -> bb1;
//     ...
//     assert(!move (_3.1: bool), "attempt to subtract with overflow") -> bb2;
//     ...
//     assert(!move (_4.1: bool), "attempt to multiply with overflow") -> bb3;
//     ...
//     assert(!move (_7.1: bool), "attempt to shift left with overflow") -> bb6;
//     ...
//     _9 = const 6i32 as u128 (Misc);
//     _7 = const compiler_builtins::int::shift::rust_u128_shlo(_1, move _9) -> bb12;
//     ...
//     assert(!move (_8.1: bool), "attempt to shift right with overflow") -> bb7;
// END rustc.test_unsigned.Lower128Bit.after.mir

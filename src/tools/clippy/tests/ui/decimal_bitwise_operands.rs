#![allow(
    clippy::erasing_op,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::unnecessary_cast,
    clippy::op_ref
)]
#![warn(clippy::decimal_bitwise_operands)]

macro_rules! bitwise_op {
    ($x:expr, $y:expr) => {
        $x & $y;
    };
}

pub const SOME_CONST: i32 = 12345;

fn main() {
    let mut x = 0;
    // BAD: Bitwise operation, decimal literal, one literal
    x & 9_8765_4321; //~ decimal_bitwise_operands
    x & 100_i32; //~ decimal_bitwise_operands
    x | (/* comment */99); //~ decimal_bitwise_operands
    x ^ (99); //~ decimal_bitwise_operands
    x &= 99; //~ decimal_bitwise_operands
    x |= { 99 }; //~ decimal_bitwise_operands
    x |= { { 99 } }; //~ decimal_bitwise_operands
    x |= {
        0b1000;
        99 //~ decimal_bitwise_operands
    };
    x ^= (99); //~ decimal_bitwise_operands

    // BAD: Bitwise operation, decimal literal, two literals
    0b1010 & 99; //~ decimal_bitwise_operands
    0b1010 | (99); //~ decimal_bitwise_operands
    0b1010 ^ (/* comment */99); //~ decimal_bitwise_operands
    99 & 0b1010; //~ decimal_bitwise_operands
    (99) | 0b1010; //~ decimal_bitwise_operands
    (/* comment */99) ^ 0b1010; //~ decimal_bitwise_operands
    0xD | { 99 }; //~ decimal_bitwise_operands
    88 & 99;
    //~^ decimal_bitwise_operands
    //~| decimal_bitwise_operands
    37 & 38 & 39;
    //~^ decimal_bitwise_operands
    //~| decimal_bitwise_operands
    //~| decimal_bitwise_operands

    // GOOD: Bitwise operation, binary/hex/octal literal, one literal
    x & 0b1010;
    x | 0b1010;
    x ^ 0b1010;
    x &= 0b1010;
    x |= 0b1010;
    x ^= 0b1010;
    x & 0xD;
    x & 0o77;
    x | 0o123;
    x ^ 0o377;
    x &= 0o777;
    x |= 0o7;
    x ^= 0o70;

    // GOOD: Bitwise operation, binary/hex/octal literal, two literals
    0b1010 & 0b1101;
    0xD ^ 0xF;
    0o377 ^ 0o77;
    0b1101 ^ 0xFF;

    // GOOD: Numeric operation, any literal
    x += 99;
    x -= 0b1010;
    x *= 0xD;
    99 + 99;
    0b1010 - 0b1101;
    0xD * 0xD;

    // BAD: Unary, cast and reference, decimal literal
    x & !100; //~ decimal_bitwise_operands
    x & -100; //~ decimal_bitwise_operands
    x & (100 as i32); //~ decimal_bitwise_operands
    x & &100; //~ decimal_bitwise_operands

    // GOOD: Unary, cast and reference, non-decimal literal
    x & !0b1101;
    x & -0xD;
    x & (0o333 as i32);
    x & &0b1010;

    // GOOD: Bitwise operation, variables only
    let y = 0;
    x & y;
    x &= y;
    x + y;
    x += y;

    // GOOD: Macro expansion (should be ignored)
    bitwise_op!(x, 123);
    bitwise_op!(0b1010, 123);

    // GOOD: Using const (should be ignored)
    x & SOME_CONST;
    x |= SOME_CONST;

    // GOOD: Parenthesized binary/hex literal (should not trigger lint)
    x & (0b1111);
    x |= (0b1010);
    x ^ (/* comment */0b1100);
    (0xFF) & x;

    // GOOD: Power of two and power of two minus one
    x & 16; // 2^4
    x | (31); // 2^5 - 1
    x ^ 0x40; // 2^6 (hex)
    x ^= 7; // 2^3 - 1

    // GOOD: Bitwise operation, single digit decimal literal
    5 & 9;
    x ^ 6;
    x ^= 7;

    // GOOD: More complex expressions
    (x + 1) & 0xFF;
    (x * 2) | (y & 0xF);
    (x ^ y) & 0b11110000;
    x | (1 << 9);

    // GOOD: Special cases
    x & 0; // All bits off
    x | !0; // All bits on
    x ^ 1; // Toggle LSB
}

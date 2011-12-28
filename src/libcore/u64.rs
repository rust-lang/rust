/*
Module: u64
*/

/*
Const: min_value

Return the minimal value for a u64
*/
const min_value: u64 = 0u64;

/*
Const: max_value

Return the maximal value for a u64
*/
const max_value: u64 = 18446744073709551615u64;

/*
Function: to_str

Convert to a string in a given base
*/
fn to_str(n: u64, radix: uint) -> str {
    assert (0u < radix && radix <= 16u);

    let r64 = radix as u64;

    fn digit(n: u64) -> str {
        ret alt n {
              0u64 { "0" }
              1u64 { "1" }
              2u64 { "2" }
              3u64 { "3" }
              4u64 { "4" }
              5u64 { "5" }
              6u64 { "6" }
              7u64 { "7" }
              8u64 { "8" }
              9u64 { "9" }
              10u64 { "a" }
              11u64 { "b" }
              12u64 { "c" }
              13u64 { "d" }
              14u64 { "e" }
              15u64 { "f" }
              _ { fail }
            };
    }

    if n == 0u64 { ret "0"; }

    let s = "";

    let n = n;
    while n > 0u64 { s = digit(n % r64) + s; n /= r64; }
    ret s;
}

/*
Function: str

Convert to a string
*/
fn str(n: u64) -> str { ret to_str(n, 10u); }

/*
Function: from_str

Parse a string as an unsigned integer.
*/
fn from_str(buf: str, radix: u64) -> u64 {
    if str::byte_len(buf) == 0u {
        #error("parse_buf(): buf is empty");
        fail;
    }
    let i = str::byte_len(buf) - 1u;
    let power = 1u64, n = 0u64;
    while true {
        let digit = char::to_digit(buf[i] as char) as u64;
        if digit >= radix { fail; }
        n += digit * power;
        power *= radix;
        if i == 0u { ret n; }
        i -= 1u;
    }
    fail;
}

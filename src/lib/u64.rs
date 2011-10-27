/*
Module: u64
*/

/*
Function: min_value

Return the minimal value for a u64
*/
pure fn min_value() -> u64 { ret 0u64; }

/*
Function: max_value

Return the maximal value for a u64
*/
pure fn max_value() -> u64 { ret 18446744073709551615u64; }

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

    while n > 0u64 { s = digit(n % r64) + s; n /= r64; }
    ret s;
}

/*
Function: str

Convert to a string
*/
fn str(n: u64) -> str { ret to_str(n, 10u); }

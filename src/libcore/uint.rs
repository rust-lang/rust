/*
Module: uint
*/

/*
Const: min_value

Return the minimal value for an uint.

This is always 0
*/
const min_value: uint = 0u;

/*
Const: max_value

Return the maximal value for an uint.

This is 2^wordsize - 1
*/
const max_value: uint = 0u - 1u;

/* Function: add */
pure fn add(x: uint, y: uint) -> uint { ret x + y; }

/* Function: sub */
pure fn sub(x: uint, y: uint) -> uint { ret x - y; }

/* Function: mul */
pure fn mul(x: uint, y: uint) -> uint { ret x * y; }

/* Function: div */
pure fn div(x: uint, y: uint) -> uint { ret x / y; }

/* Function: div_ceil

   Divide two numbers, return the result, rounded up.

   Parameters:
   x - an integer
   y - an integer distinct from 0u

   Return:
    The smallest integer `q` such that `x/y <= q`.
*/
pure fn div_ceil(x: uint, y: uint) -> uint {
    let div = div(x, y);
    if x % y == 0u { ret div;}
    else { ret div + 1u; }
}

/* Function: div_ceil

   Divide two numbers, return the result, rounded to the closest integer.

   Parameters:
   x - an integer
   y - an integer distinct from 0u

   Return:
    The integer `q` closest to `x/y`.
*/
pure fn div_round(x: uint, y: uint) -> uint {
    let div = div(x, y);
    if x % y * 2u  < y { ret div;}
    else { ret div + 1u; }
}

/* Function: div_ceil

   Divide two numbers, return the result, rounded down.

   Parameters:
   x - an integer
   y - an integer distinct from 0u

   Note: This is the same function as `div`.

   Return:
    The smallest integer `q` such that `x/y <= q`. This
   is either `x/y` or `x/y + 1`.
*/
pure fn div_floor(x: uint, y: uint) -> uint { ret x / y; }

/* Function: rem */
pure fn rem(x: uint, y: uint) -> uint { ret x % y; }

/* Predicate: lt */
pure fn lt(x: uint, y: uint) -> bool { ret x < y; }

/* Predicate: le */
pure fn le(x: uint, y: uint) -> bool { ret x <= y; }

/* Predicate: eq */
pure fn eq(x: uint, y: uint) -> bool { ret x == y; }

/* Predicate: ne */
pure fn ne(x: uint, y: uint) -> bool { ret x != y; }

/* Predicate: ge */
pure fn ge(x: uint, y: uint) -> bool { ret x >= y; }

/* Predicate: gt */
pure fn gt(x: uint, y: uint) -> bool { ret x > y; }

/*
Function: hash

Produce a uint suitable for use in a hash table
*/
fn hash(x: uint) -> uint { ret x; }

/*
Function: range

Iterate over the range [`lo`..`hi`)
*/
fn range(lo: uint, hi: uint, it: fn(uint)) {
    let i = lo;
    while i < hi { it(i); i += 1u; }
}

/*
Function: loop

Iterate over the range [`lo`..`hi`), or stop when requested

Parameters:
lo - The integer at which to start the loop (included)
hi - The integer at which to stop the loop (excluded)
it - A block to execute with each consecutive integer of the range.
Return `true` to continue, `false` to stop.

Returns:

`true` If execution proceeded correctly, `false` if it was interrupted,
that is if `it` returned `false` at any point.
*/
fn loop(lo: uint, hi: uint, it: fn(uint) -> bool) -> bool {
    let i = lo;
    while i < hi {
        if (!it(i)) { ret false; }
        i += 1u;
    }
    ret true;
}

/*
Function: next_power_of_two

Returns the smallest power of 2 greater than or equal to `n`
*/
fn next_power_of_two(n: uint) -> uint {
    let halfbits: uint = sys::size_of::<uint>() * 4u;
    let tmp: uint = n - 1u;
    let shift: uint = 1u;
    while shift <= halfbits { tmp |= tmp >> shift; shift <<= 1u; }
    ret tmp + 1u;
}

/*
Function: parse_buf

Parse a buffer of bytes

Parameters:

buf - A byte buffer
radix - The base of the number

Failure:

buf must not be empty
*/
fn parse_buf(buf: [u8], radix: uint) -> uint {
    if vec::len::<u8>(buf) == 0u {
        #error("parse_buf(): buf is empty");
        fail;
    }
    let i = vec::len::<u8>(buf) - 1u;
    let power = 1u;
    let n = 0u;
    while true {
        let digit = char::to_digit(buf[i] as char);
        if (digit as uint) >= radix {
            fail;
        }
        n += (digit as uint) * power;
        power *= radix;
        if i == 0u { ret n; }
        i -= 1u;
    }
    fail;
}

/*
Function: from_str

Parse a string to an int

Failure:

s must not be empty
*/
fn from_str(s: str) -> uint { parse_buf(str::bytes(s), 10u) }

/*
Function: to_str

Convert to a string in a given base
*/
fn to_str(num: uint, radix: uint) -> str {
    let n = num;
    assert (0u < radix && radix <= 16u);
    fn digit(n: uint) -> char {
        ret alt n {
              0u { '0' }
              1u { '1' }
              2u { '2' }
              3u { '3' }
              4u { '4' }
              5u { '5' }
              6u { '6' }
              7u { '7' }
              8u { '8' }
              9u { '9' }
              10u { 'a' }
              11u { 'b' }
              12u { 'c' }
              13u { 'd' }
              14u { 'e' }
              15u { 'f' }
              _ { fail }
            };
    }
    if n == 0u { ret "0"; }
    let s: str = "";
    while n != 0u {
        s += str::from_byte(digit(n % radix) as u8);
        n /= radix;
    }
    let s1: str = "";
    let len: uint = str::byte_len(s);
    while len != 0u { len -= 1u; s1 += str::from_byte(s[len]); }
    ret s1;
}

/*
Function: str

Convert to a string
*/
fn str(i: uint) -> str { ret to_str(i, 10u); }

#[cfg(test)]
mod tests {

    #[test]
    fn test_from_str() {
        assert (uint::from_str("0") == 0u);
        assert (uint::from_str("3") == 3u);
        assert (uint::from_str("10") == 10u);
        assert (uint::from_str("123456789") == 123456789u);
        assert (uint::from_str("00100") == 100u);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_from_str_fail_1() {
        uint::from_str(" ");
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_from_str_fail_2() {
        uint::from_str("x");
    }

    #[test]
    fn test_parse_buf() {
        import str::bytes;
        assert (uint::parse_buf(bytes("123"), 10u) == 123u);
        assert (uint::parse_buf(bytes("1001"), 2u) == 9u);
        assert (uint::parse_buf(bytes("123"), 8u) == 83u);
        assert (uint::parse_buf(bytes("123"), 16u) == 291u);
        assert (uint::parse_buf(bytes("ffff"), 16u) == 65535u);
        assert (uint::parse_buf(bytes("z"), 36u) == 35u);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_parse_buf_fail_1() {
        uint::parse_buf(str::bytes("Z"), 10u);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_parse_buf_fail_2() {
        uint::parse_buf(str::bytes("_"), 2u);
    }

    #[test]
    fn test_next_power_of_two() {
        assert (uint::next_power_of_two(0u) == 0u);
        assert (uint::next_power_of_two(1u) == 1u);
        assert (uint::next_power_of_two(2u) == 2u);
        assert (uint::next_power_of_two(3u) == 4u);
        assert (uint::next_power_of_two(4u) == 4u);
        assert (uint::next_power_of_two(5u) == 8u);
        assert (uint::next_power_of_two(6u) == 8u);
        assert (uint::next_power_of_two(7u) == 8u);
        assert (uint::next_power_of_two(8u) == 8u);
        assert (uint::next_power_of_two(9u) == 16u);
        assert (uint::next_power_of_two(10u) == 16u);
        assert (uint::next_power_of_two(11u) == 16u);
        assert (uint::next_power_of_two(12u) == 16u);
        assert (uint::next_power_of_two(13u) == 16u);
        assert (uint::next_power_of_two(14u) == 16u);
        assert (uint::next_power_of_two(15u) == 16u);
        assert (uint::next_power_of_two(16u) == 16u);
        assert (uint::next_power_of_two(17u) == 32u);
        assert (uint::next_power_of_two(18u) == 32u);
        assert (uint::next_power_of_two(19u) == 32u);
        assert (uint::next_power_of_two(20u) == 32u);
        assert (uint::next_power_of_two(21u) == 32u);
        assert (uint::next_power_of_two(22u) == 32u);
        assert (uint::next_power_of_two(23u) == 32u);
        assert (uint::next_power_of_two(24u) == 32u);
        assert (uint::next_power_of_two(25u) == 32u);
        assert (uint::next_power_of_two(26u) == 32u);
        assert (uint::next_power_of_two(27u) == 32u);
        assert (uint::next_power_of_two(28u) == 32u);
        assert (uint::next_power_of_two(29u) == 32u);
        assert (uint::next_power_of_two(30u) == 32u);
        assert (uint::next_power_of_two(31u) == 32u);
        assert (uint::next_power_of_two(32u) == 32u);
        assert (uint::next_power_of_two(33u) == 64u);
        assert (uint::next_power_of_two(34u) == 64u);
        assert (uint::next_power_of_two(35u) == 64u);
        assert (uint::next_power_of_two(36u) == 64u);
        assert (uint::next_power_of_two(37u) == 64u);
        assert (uint::next_power_of_two(38u) == 64u);
        assert (uint::next_power_of_two(39u) == 64u);
    }

    #[test]
    fn test_overflows() {
        assert (uint::max_value > 0u);
        assert (uint::min_value <= 0u);
        assert (uint::min_value + uint::max_value + 1u == 0u);
    }

    #[test]
    fn test_div() {
        assert(uint::div_floor(3u, 4u) == 0u);
        assert(uint::div_ceil(3u, 4u)  == 1u);
        assert(uint::div_round(3u, 4u) == 1u);
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

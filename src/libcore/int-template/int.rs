type T = int;

#[cfg(target_arch = "x86")]
const bits: T = 32 as T;

#[cfg(target_arch = "x86_64")]
const bits: T = 64 as T;

#[doc = "Produce a uint suitable for use in a hash table"]
pure fn hash(x: int) -> uint { ret x as uint; }

#[doc = "
Parse a buffer of bytes

# Arguments

* buf - A byte buffer
* radix - The base of the number
"]
fn parse_buf(buf: [u8], radix: uint) -> option<int> {
    if vec::len(buf) == 0u { ret none; }
    let mut i = vec::len(buf) - 1u;
    let mut start = 0u;
    let mut power = 1;

    if buf[0] == ('-' as u8) {
        power = -1;
        start = 1u;
    }
    let mut n = 0;
    loop {
        alt char::to_digit(buf[i] as char, radix) {
          some(d) { n += (d as int) * power; }
          none { ret none; }
        }
        power *= radix as int;
        if i <= start { ret some(n); }
        i -= 1u;
    };
}

#[doc = "Parse a string to an int"]
fn from_str(s: str) -> option<int> { parse_buf(str::bytes(s), 10u) }

#[doc = "Convert to a string in a given base"]
fn to_str(n: int, radix: uint) -> str {
    assert (0u < radix && radix <= 16u);
    ret if n < 0 {
            "-" + uint::to_str(-n as uint, radix)
        } else { uint::to_str(n as uint, radix) };
}

#[doc = "Convert to a string"]
fn str(i: int) -> str { ret to_str(i, 10u); }

#[doc = "Returns `base` raised to the power of `exponent`"]
fn pow(base: int, exponent: uint) -> int {
    if exponent == 0u { ret 1; } //Not mathemtically true if [base == 0]
    if base     == 0  { ret 0; }
    let mut my_pow  = exponent;
    let mut acc     = 1;
    let mut multiplier = base;
    while(my_pow > 0u) {
      if my_pow % 2u == 1u {
         acc *= multiplier;
      }
      my_pow     /= 2u;
      multiplier *= multiplier;
    }
    ret acc;
}


#[test]
fn test_from_str() {
    assert from_str("0") == some(0);
    assert from_str("3") == some(3);
    assert from_str("10") == some(10);
    assert from_str("123456789") == some(123456789);
    assert from_str("00100") == some(100);

    assert from_str("-1") == some(-1);
    assert from_str("-3") == some(-3);
    assert from_str("-10") == some(-10);
    assert from_str("-123456789") == some(-123456789);
    assert from_str("-00100") == some(-100);

    assert from_str(" ") == none;
    assert from_str("x") == none;
}

#[test]
fn test_parse_buf() {
    import str::bytes;
    assert parse_buf(bytes("123"), 10u) == some(123);
    assert parse_buf(bytes("1001"), 2u) == some(9);
    assert parse_buf(bytes("123"), 8u) == some(83);
    assert parse_buf(bytes("123"), 16u) == some(291);
    assert parse_buf(bytes("ffff"), 16u) == some(65535);
    assert parse_buf(bytes("FFFF"), 16u) == some(65535);
    assert parse_buf(bytes("z"), 36u) == some(35);
    assert parse_buf(bytes("Z"), 36u) == some(35);

    assert parse_buf(bytes("-123"), 10u) == some(-123);
    assert parse_buf(bytes("-1001"), 2u) == some(-9);
    assert parse_buf(bytes("-123"), 8u) == some(-83);
    assert parse_buf(bytes("-123"), 16u) == some(-291);
    assert parse_buf(bytes("-ffff"), 16u) == some(-65535);
    assert parse_buf(bytes("-FFFF"), 16u) == some(-65535);
    assert parse_buf(bytes("-z"), 36u) == some(-35);
    assert parse_buf(bytes("-Z"), 36u) == some(-35);

    assert parse_buf(str::bytes("Z"), 35u) == none;
    assert parse_buf(str::bytes("-9"), 2u) == none;
}

#[test]
fn test_to_str() {
    import str::eq;
    assert (eq(to_str(0, 10u), "0"));
    assert (eq(to_str(1, 10u), "1"));
    assert (eq(to_str(-1, 10u), "-1"));
    assert (eq(to_str(255, 16u), "ff"));
    assert (eq(to_str(100, 10u), "100"));
}

#[test]
fn test_pow() {
    assert (pow(0, 0u) == 1);
    assert (pow(0, 1u) == 0);
    assert (pow(0, 2u) == 0);
    assert (pow(-1, 0u) == 1);
    assert (pow(1, 0u) == 1);
    assert (pow(-3, 2u) == 9);
    assert (pow(-3, 3u) == -27);
    assert (pow(4, 9u) == 262144);
}

#[test]
fn test_overflows() {
   assert (max_value > 0);
   assert (min_value <= 0);
   assert (min_value + max_value + 1 == 0);
}

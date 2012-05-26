import T = inst::T;

export min_value, max_value;
export min, max;
export add, sub, mul, div, rem;
export lt, le, eq, ne, ge, gt;
export is_positive, is_negative;
export is_nonpositive, is_nonnegative;
export range;
export compl;
export to_str, from_str, from_str_radix, str, parse_buf;

const min_value: T = 0 as T;
const max_value: T = 0 as T - 1 as T;

pure fn min(x: T, y: T) -> T { if x < y { x } else { y } }
pure fn max(x: T, y: T) -> T { if x > y { x } else { y } }

pure fn add(x: T, y: T) -> T { x + y }
pure fn sub(x: T, y: T) -> T { x - y }
pure fn mul(x: T, y: T) -> T { x * y }
pure fn div(x: T, y: T) -> T { x / y }
pure fn rem(x: T, y: T) -> T { x % y }

pure fn lt(x: T, y: T) -> bool { x < y }
pure fn le(x: T, y: T) -> bool { x <= y }
pure fn eq(x: T, y: T) -> bool { x == y }
pure fn ne(x: T, y: T) -> bool { x != y }
pure fn ge(x: T, y: T) -> bool { x >= y }
pure fn gt(x: T, y: T) -> bool { x > y }

pure fn is_positive(x: T) -> bool { x > 0 as T }
pure fn is_negative(x: T) -> bool { x < 0 as T }
pure fn is_nonpositive(x: T) -> bool { x <= 0 as T }
pure fn is_nonnegative(x: T) -> bool { x >= 0 as T }

#[doc = "Iterate over the range [`lo`..`hi`)"]
fn range(lo: T, hi: T, it: fn(T) -> bool) {
    let mut i = lo;
    while i < hi {
        if !it(i) { break }
        i += 1 as T;
    }
}

#[doc = "Computes the bitwise complement"]
pure fn compl(i: T) -> T {
    max_value ^ i
}

#[doc = "
Parse a buffer of bytes

# Arguments

* buf - A byte buffer
* radix - The base of the number

# Failure

`buf` must not be empty
"]
fn parse_buf(buf: [u8], radix: uint) -> option<T> {
    if vec::len(buf) == 0u { ret none; }
    let mut i = vec::len(buf) - 1u;
    let mut power = 1u as T;
    let mut n = 0u as T;
    loop {
        alt char::to_digit(buf[i] as char, radix) {
          some(d) { n += d as T * power; }
          none { ret none; }
        }
        power *= radix as T;
        if i == 0u { ret some(n); }
        i -= 1u;
    };
}

#[doc = "Parse a string to an int"]
fn from_str(s: str) -> option<T> { parse_buf(str::bytes(s), 10u) }

#[doc = "Parse a string as an unsigned integer."]
fn from_str_radix(buf: str, radix: u64) -> option<u64> {
    if str::len(buf) == 0u { ret none; }
    let mut i = str::len(buf) - 1u;
    let mut power = 1u64, n = 0u64;
    loop {
        alt char::to_digit(buf[i] as char, radix as uint) {
          some(d) { n += d as u64 * power; }
          none { ret none; }
        }
        power *= radix;
        if i == 0u { ret some(n); }
        i -= 1u;
    };
}

#[doc = "Convert to a string in a given base"]
fn to_str(num: T, radix: uint) -> str {
    assert (0u < radix && radix <= 16u);
    let mut n = num;
    let radix = radix as T;
    fn digit(n: T) -> char {
        ret alt n {
              0u as T { '0' }
              1u as T { '1' }
              2u as T { '2' }
              3u as T { '3' }
              4u as T { '4' }
              5u as T { '5' }
              6u as T { '6' }
              7u as T { '7' }
              8u as T { '8' }
              9u as T { '9' }
              10u as T { 'a' }
              11u as T { 'b' }
              12u as T { 'c' }
              13u as T { 'd' }
              14u as T { 'e' }
              15u as T { 'f' }
              _ { fail }
            };
    }
    if n == 0u as T { ret "0"; }
    let mut s: str = "";
    while n != 0u as T {
        s += str::from_byte(digit(n % radix) as u8);
        n /= radix;
    }
    let mut s1: str = "";
    let mut len: uint = str::len(s);
    while len != 0u { len -= 1u; s1 += str::from_byte(s[len]); }
    ret s1;
}

#[doc = "Convert to a string"]
fn str(i: T) -> str { ret to_str(i, 10u); }

#[test]
#[ignore]
fn test_from_str() {
    assert from_str("0") == some(0u as T);
    assert from_str("3") == some(3u as T);
    assert from_str("10") == some(10u as T);
    assert from_str("123456789") == some(123456789u as T);
    assert from_str("00100") == some(100u as T);

    assert from_str("") == none;
    assert from_str(" ") == none;
    assert from_str("x") == none;
}

#[test]
#[ignore]
fn test_parse_buf() {
    import str::bytes;
    assert parse_buf(bytes("123"), 10u) == some(123u as T);
    assert parse_buf(bytes("1001"), 2u) == some(9u as T);
    assert parse_buf(bytes("123"), 8u) == some(83u as T);
    assert parse_buf(bytes("123"), 16u) == some(291u as T);
    assert parse_buf(bytes("ffff"), 16u) == some(65535u as T);
    assert parse_buf(bytes("z"), 36u) == some(35u as T);

    assert parse_buf(str::bytes("Z"), 10u) == none;
    assert parse_buf(str::bytes("_"), 2u) == none;
}

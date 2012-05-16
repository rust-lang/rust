import T = inst::T;

export min_value, max_value;
export min, max;
export add, sub, mul, div, rem;
export lt, le, eq, ne, ge, gt;
export is_positive, is_negative;
export is_nonpositive, is_nonnegative;
export range;
export compl;
export abs;
export parse_buf, from_str, to_str, str;

const min_value: T = -1 as T << (inst::bits - 1 as T);
const max_value: T = min_value - 1 as T;

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
fn range(lo: T, hi: T, it: fn(T)) {
    let mut i = lo;
    while i < hi { it(i); i += 1 as T; }
}

#[doc = "Computes the bitwise complement"]
pure fn compl(i: T) -> T {
    -1 as T ^ i
}

#[doc = "Computes the absolute value"]
// FIXME: abs should return an unsigned int (#2353)
pure fn abs(i: T) -> T {
    if is_negative(i) { -i } else { i }
}

#[doc = "
Parse a buffer of bytes

# Arguments

* buf - A byte buffer
* radix - The base of the number
"]
fn parse_buf(buf: [u8], radix: uint) -> option<T> {
    if vec::len(buf) == 0u { ret none; }
    let mut i = vec::len(buf) - 1u;
    let mut start = 0u;
    let mut power = 1 as T;

    if buf[0] == ('-' as u8) {
        power = -1 as T;
        start = 1u;
    }
    let mut n = 0 as T;
    loop {
        alt char::to_digit(buf[i] as char, radix) {
          some(d) { n += (d as T) * power; }
          none { ret none; }
        }
        power *= radix as T;
        if i <= start { ret some(n); }
        i -= 1u;
    };
}

#[doc = "Parse a string to an int"]
fn from_str(s: str) -> option<T> { parse_buf(str::bytes(s), 10u) }

#[doc = "Convert to a string in a given base"]
fn to_str(n: T, radix: uint) -> str {
    assert (0u < radix && radix <= 16u);
    ret if n < 0 as T {
            "-" + uint::to_str(-n as uint, radix)
        } else { uint::to_str(n as uint, radix) };
}

#[doc = "Convert to a string"]
fn str(i: T) -> str { ret to_str(i, 10u); }


#[test]
fn test_from_str() {
    assert from_str("0") == some(0 as T);
    assert from_str("3") == some(3 as T);
    assert from_str("10") == some(10 as T);
    assert from_str("123456789") == some(123456789 as T);
    assert from_str("00100") == some(100 as T);

    assert from_str("-1") == some(-1 as T);
    assert from_str("-3") == some(-3 as T);
    assert from_str("-10") == some(-10 as T);
    assert from_str("-123456789") == some(-123456789 as T);
    assert from_str("-00100") == some(-100 as T);

    assert from_str(" ") == none;
    assert from_str("x") == none;
}

#[test]
fn test_parse_buf() {
    import str::bytes;
    assert parse_buf(bytes("123"), 10u) == some(123 as T);
    assert parse_buf(bytes("1001"), 2u) == some(9 as T);
    assert parse_buf(bytes("123"), 8u) == some(83 as T);
    assert parse_buf(bytes("123"), 16u) == some(291 as T);
    assert parse_buf(bytes("ffff"), 16u) == some(65535 as T);
    assert parse_buf(bytes("FFFF"), 16u) == some(65535 as T);
    assert parse_buf(bytes("z"), 36u) == some(35 as T);
    assert parse_buf(bytes("Z"), 36u) == some(35 as T);

    assert parse_buf(bytes("-123"), 10u) == some(-123 as T);
    assert parse_buf(bytes("-1001"), 2u) == some(-9 as T);
    assert parse_buf(bytes("-123"), 8u) == some(-83 as T);
    assert parse_buf(bytes("-123"), 16u) == some(-291 as T);
    assert parse_buf(bytes("-ffff"), 16u) == some(-65535 as T);
    assert parse_buf(bytes("-FFFF"), 16u) == some(-65535 as T);
    assert parse_buf(bytes("-z"), 36u) == some(-35 as T);
    assert parse_buf(bytes("-Z"), 36u) == some(-35 as T);

    assert parse_buf(str::bytes("Z"), 35u) == none;
    assert parse_buf(str::bytes("-9"), 2u) == none;
}

#[test]
fn test_to_str() {
    import str::eq;
    assert (eq(to_str(0 as T, 10u), "0"));
    assert (eq(to_str(1 as T, 10u), "1"));
    assert (eq(to_str(-1 as T, 10u), "-1"));
    assert (eq(to_str(127 as T, 16u), "7f"));
    assert (eq(to_str(100 as T, 10u), "100"));
}

import T = inst::T;
import cmp::{eq, ord};

export min_value, max_value;
export min, max;
export add, sub, mul, div, rem;
export lt, le, eq, ne, ge, gt;
export is_positive, is_negative;
export is_nonpositive, is_nonnegative;
export range;
export compl;
export abs;
export parse_buf, from_str, to_str, to_str_bytes, str;
export num, ord, eq, times, timesi;
export bits, bytes;

const bits : uint = inst::bits;
const bytes : uint = (inst::bits / 8);

const min_value: T = (-1 as T) << (bits - 1);
const max_value: T = min_value - 1 as T;

pure fn min(&&x: T, &&y: T) -> T { if x < y { x } else { y } }
pure fn max(&&x: T, &&y: T) -> T { if x > y { x } else { y } }

pure fn add(x: &T, y: &T) -> T { *x + *y }
pure fn sub(x: &T, y: &T) -> T { *x - *y }
pure fn mul(x: &T, y: &T) -> T { *x * *y }
pure fn div(x: &T, y: &T) -> T { *x / *y }
pure fn rem(x: &T, y: &T) -> T { *x % *y }

pure fn lt(x: &T, y: &T) -> bool { *x < *y }
pure fn le(x: &T, y: &T) -> bool { *x <= *y }
pure fn eq(x: &T, y: &T) -> bool { *x == *y }
pure fn ne(x: &T, y: &T) -> bool { *x != *y }
pure fn ge(x: &T, y: &T) -> bool { *x >= *y }
pure fn gt(x: &T, y: &T) -> bool { *x > *y }

pure fn is_positive(x: T) -> bool { x > 0 as T }
pure fn is_negative(x: T) -> bool { x < 0 as T }
pure fn is_nonpositive(x: T) -> bool { x <= 0 as T }
pure fn is_nonnegative(x: T) -> bool { x >= 0 as T }

#[inline(always)]
/// Iterate over the range [`lo`..`hi`)
fn range(lo: T, hi: T, it: fn(T) -> bool) {
    let mut i = lo;
    while i < hi {
        if !it(i) { break }
        i += 1 as T;
    }
}

/// Computes the bitwise complement
pure fn compl(i: T) -> T {
    -1 as T ^ i
}

/// Computes the absolute value
// FIXME: abs should return an unsigned int (#2353)
pure fn abs(i: T) -> T {
    if is_negative(i) { -i } else { i }
}

impl ord of ord for T {
    pure fn lt(&&other: T) -> bool {
        return self < other;
    }
}

impl eq of eq for T {
    pure fn eq(&&other: T) -> bool {
        return self == other;
    }
}


impl num of num::num for T {
    pure fn add(&&other: T)    -> T { return self + other; }
    pure fn sub(&&other: T)    -> T { return self - other; }
    pure fn mul(&&other: T)    -> T { return self * other; }
    pure fn div(&&other: T)    -> T { return self / other; }
    pure fn modulo(&&other: T) -> T { return self % other; }
    pure fn neg()              -> T { return -self;        }

    pure fn to_int()         -> int { return self as int; }
    pure fn from_int(n: int) -> T   { return n as T;      }
}

impl times of iter::times for T {
    #[inline(always)]
    #[doc = "A convenience form for basic iteration. Given a variable `x` \
        of any numeric type, the expression `for x.times { /* anything */ }` \
        will execute the given function exactly x times. If we assume that \
        `x` is an int, this is functionally equivalent to \
        `for int::range(0, x) |_i| { /* anything */ }`."]
    fn times(it: fn() -> bool) {
        if self < 0 {
            fail fmt!{"The .times method expects a nonnegative number, \
                       but found %?", self};
        }
        let mut i = self;
        while i > 0 {
            if !it() { break }
            i -= 1;
        }
    }
}

impl timesi of iter::timesi for T {
    #[inline(always)]
    /// Like `times`, but provides an index
    fn timesi(it: fn(uint) -> bool) {
        let slf = self as uint;
        if slf < 0u {
            fail fmt!{"The .timesi method expects a nonnegative number, \
                       but found %?", self};
        }
        let mut i = 0u;
        while i < slf {
            if !it(i) { break }
            i += 1u;
        }
    }
}

/**
 * Parse a buffer of bytes
 *
 * # Arguments
 *
 * * buf - A byte buffer
 * * radix - The base of the number
 */
fn parse_buf(buf: ~[u8], radix: uint) -> option<T> {
    if vec::len(buf) == 0u { return none; }
    let mut i = vec::len(buf) - 1u;
    let mut start = 0u;
    let mut power = 1 as T;

    if buf[0] == ('-' as u8) {
        power = -1 as T;
        start = 1u;
    }
    let mut n = 0 as T;
    loop {
        match char::to_digit(buf[i] as char, radix) {
          some(d) => n += (d as T) * power,
          none => return none
        }
        power *= radix as T;
        if i <= start { return some(n); }
        i -= 1u;
    };
}

/// Parse a string to an int
fn from_str(s: ~str) -> option<T> { parse_buf(str::bytes(s), 10u) }

/// Convert to a string in a given base
fn to_str(n: T, radix: uint) -> ~str {
    do to_str_bytes(n, radix) |slice| {
        do vec::as_buf(slice) |p, len| {
            unsafe { str::unsafe::from_buf_len(p, len) }
        }
    }
}

fn to_str_bytes<U>(n: T, radix: uint, f: fn(v: &[u8]) -> U) -> U {
    if n < 0 as T {
        uint::to_str_bytes(true, -n as uint, radix, f)
    } else {
        uint::to_str_bytes(false, n as uint, radix, f)
    }
}

/// Convert to a string
fn str(i: T) -> ~str { return to_str(i, 10u); }

// FIXME: Has alignment issues on windows and 32-bit linux (#2609)
#[test]
#[ignore]
fn test_from_str() {
    assert from_str(~"0") == some(0 as T);
    assert from_str(~"3") == some(3 as T);
    assert from_str(~"10") == some(10 as T);
    assert from_str(~"123456789") == some(123456789 as T);
    assert from_str(~"00100") == some(100 as T);

    assert from_str(~"-1") == some(-1 as T);
    assert from_str(~"-3") == some(-3 as T);
    assert from_str(~"-10") == some(-10 as T);
    assert from_str(~"-123456789") == some(-123456789 as T);
    assert from_str(~"-00100") == some(-100 as T);

    assert from_str(~" ") == none;
    assert from_str(~"x") == none;
}

// FIXME: Has alignment issues on windows and 32-bit linux (#2609)
#[test]
#[ignore]
fn test_parse_buf() {
    import str::bytes;
    assert parse_buf(bytes(~"123"), 10u) == some(123 as T);
    assert parse_buf(bytes(~"1001"), 2u) == some(9 as T);
    assert parse_buf(bytes(~"123"), 8u) == some(83 as T);
    assert parse_buf(bytes(~"123"), 16u) == some(291 as T);
    assert parse_buf(bytes(~"ffff"), 16u) == some(65535 as T);
    assert parse_buf(bytes(~"FFFF"), 16u) == some(65535 as T);
    assert parse_buf(bytes(~"z"), 36u) == some(35 as T);
    assert parse_buf(bytes(~"Z"), 36u) == some(35 as T);

    assert parse_buf(bytes(~"-123"), 10u) == some(-123 as T);
    assert parse_buf(bytes(~"-1001"), 2u) == some(-9 as T);
    assert parse_buf(bytes(~"-123"), 8u) == some(-83 as T);
    assert parse_buf(bytes(~"-123"), 16u) == some(-291 as T);
    assert parse_buf(bytes(~"-ffff"), 16u) == some(-65535 as T);
    assert parse_buf(bytes(~"-FFFF"), 16u) == some(-65535 as T);
    assert parse_buf(bytes(~"-z"), 36u) == some(-35 as T);
    assert parse_buf(bytes(~"-Z"), 36u) == some(-35 as T);

    assert parse_buf(str::bytes(~"Z"), 35u) == none;
    assert parse_buf(str::bytes(~"-9"), 2u) == none;
}

#[test]
fn test_to_str() {
    assert (to_str(0 as T, 10u) == ~"0");
    assert (to_str(1 as T, 10u) == ~"1");
    assert (to_str(-1 as T, 10u) == ~"-1");
    assert (to_str(127 as T, 16u) == ~"7f");
    assert (to_str(100 as T, 10u) == ~"100");
}

#[test]
fn test_interfaces() {
    fn test<U:num::num>(ten: U) {
        assert (ten.to_int() == 10);

        let two = ten.from_int(2);
        assert (two.to_int() == 2);

        assert (ten.add(two) == ten.from_int(12));
        assert (ten.sub(two) == ten.from_int(8));
        assert (ten.mul(two) == ten.from_int(20));
        assert (ten.div(two) == ten.from_int(5));
        assert (ten.modulo(two) == ten.from_int(0));
        assert (ten.neg() == ten.from_int(-10));
    }

    test(10 as T);
}

#[test]
fn test_times() {
    import iter::times;
    let ten = 10 as T;
    let mut accum = 0;
    for ten.times { accum += 1; }
    assert (accum == 10);
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_times_negative() {
    import iter::times;
    for (-10).times { log(error, ~"nope!"); }
}

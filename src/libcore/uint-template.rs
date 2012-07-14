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
export to_str, to_str_bytes;
export from_str, from_str_radix, str, parse_buf;
export num, ord, eq, times;

const min_value: T = 0 as T;
const max_value: T = 0 as T - 1 as T;

pure fn min(&&x: T, &&y: T) -> T { if x < y { x } else { y } }
pure fn max(&&x: T, &&y: T) -> T { if x > y { x } else { y } }

pure fn add(&&x: T, &&y: T) -> T { x + y }
pure fn sub(&&x: T, &&y: T) -> T { x - y }
pure fn mul(&&x: T, &&y: T) -> T { x * y }
pure fn div(&&x: T, &&y: T) -> T { x / y }
pure fn rem(&&x: T, &&y: T) -> T { x % y }

pure fn lt(&&x: T, &&y: T) -> bool { x < y }
pure fn le(&&x: T, &&y: T) -> bool { x <= y }
pure fn eq(&&x: T, &&y: T) -> bool { x == y }
pure fn ne(&&x: T, &&y: T) -> bool { x != y }
pure fn ge(&&x: T, &&y: T) -> bool { x >= y }
pure fn gt(&&x: T, &&y: T) -> bool { x > y }

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
    max_value ^ i
}

impl ord of ord for T {
    fn lt(&&other: T) -> bool {
        ret self < other;
    }
}

impl eq of eq for T {
    fn eq(&&other: T) -> bool {
        ret self == other;
    }
}

impl num of num::num for T {
    fn add(&&other: T)    -> T { ret self + other; }
    fn sub(&&other: T)    -> T { ret self - other; }
    fn mul(&&other: T)    -> T { ret self * other; }
    fn div(&&other: T)    -> T { ret self / other; }
    fn modulo(&&other: T) -> T { ret self % other; }
    fn neg()              -> T { ret -self;        }

    fn to_int()         -> int { ret self as int; }
    fn from_int(n: int) -> T   { ret n as T;      }
}

/**
 * Parse a buffer of bytes
 *
 * # Arguments
 *
 * * buf - A byte buffer
 * * radix - The base of the number
 *
 * # Failure
 *
 * `buf` must not be empty
 */
fn parse_buf(buf: ~[u8], radix: uint) -> option<T> {
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

impl times of iter::times for T {
    #[inline(always)]
    #[doc = "A convenience form for basic iteration. Given a variable `x` \
        of any numeric type, the expression `for x.times { /* anything */ }` \
        will execute the given function exactly x times. If we assume that \
        `x` is an int, this is functionally equivalent to \
        `for int::range(0, x) |_i| { /* anything */ }`."]
    fn times(it: fn() -> bool) {
        let mut i = self;
        while i > 0 {
            if !it() { break }
            i -= 1;
        }
    }
}

/// Parse a string to an int
fn from_str(s: ~str) -> option<T> { parse_buf(str::bytes(s), 10u) }

/// Parse a string as an unsigned integer.
fn from_str_radix(buf: ~str, radix: u64) -> option<u64> {
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

/**
 * Convert to a string in a given base
 *
 * # Failure
 *
 * Fails if `radix` < 2 or `radix` > 16
 */
fn to_str(num: T, radix: uint) -> ~str {
    do to_str_bytes(false, num, radix) |slice| {
        do vec::unpack_slice(slice) |p, len| {
            unsafe { str::unsafe::from_buf_len(p, len) }
        }
    }
}

/// Low-level helper routine for string conversion.
fn to_str_bytes<U>(neg: bool, num: T, radix: uint,
                   f: fn(v: &[u8]) -> U) -> U {

    #[inline(always)]
    fn digit(n: T) -> u8 {
        if n <= 9u as T {
            n as u8 + '0' as u8
        } else if n <= 15u as T {
            (n - 10 as T) as u8 + 'a' as u8
        } else {
            fail;
        }
    }

    assert (1u < radix && radix <= 16u);

    // Enough room to hold any number in any radix.
    // Worst case: 64-bit number, binary-radix, with
    // a leading negative sign = 65 bytes.
    let buf : [mut u8]/65 =
        [mut
         0u8,0u8,0u8,0u8,0u8, 0u8,0u8,0u8,0u8,0u8,
         0u8,0u8,0u8,0u8,0u8, 0u8,0u8,0u8,0u8,0u8,

         0u8,0u8,0u8,0u8,0u8, 0u8,0u8,0u8,0u8,0u8,
         0u8,0u8,0u8,0u8,0u8, 0u8,0u8,0u8,0u8,0u8,

         0u8,0u8,0u8,0u8,0u8, 0u8,0u8,0u8,0u8,0u8,
         0u8,0u8,0u8,0u8,0u8, 0u8,0u8,0u8,0u8,0u8,

         0u8,0u8,0u8,0u8,0u8
         ]/65;

    // FIXME (#2649): post-snapshot, you can do this without the raw
    // pointers and unsafe bits, and the codegen will prove it's all
    // in-bounds, no extra cost.

    unsafe {
        do vec::unpack_slice(buf) |p, len| {
            let mp = p as *mut u8;
            let mut i = len;
            let mut n = num;
            let radix = radix as T;
            loop {
                i -= 1u;
                assert 0u < i && i < len;
                *ptr::mut_offset(mp, i) = digit(n % radix);
                n /= radix;
                if n == 0 as T { break; }
            }

            assert 0u < i && i < len;

            if neg {
                i -= 1u;
                *ptr::mut_offset(mp, i) = '-' as u8;
            }

            vec::unsafe::form_slice(ptr::offset(p, i),
                                    len - i, f)
        }
    }
}

/// Convert to a string
fn str(i: T) -> ~str { ret to_str(i, 10u); }

#[test]
fn test_to_str() {
    assert to_str(0 as T, 10u) == ~"0";
    assert to_str(1 as T, 10u) == ~"1";
    assert to_str(2 as T, 10u) == ~"2";
    assert to_str(11 as T, 10u) == ~"11";
    assert to_str(11 as T, 16u) == ~"b";
    assert to_str(255 as T, 16u) == ~"ff";
    assert to_str(0xff as T, 10u) == ~"255";
}

#[test]
#[ignore]
fn test_from_str() {
    assert from_str(~"0") == some(0u as T);
    assert from_str(~"3") == some(3u as T);
    assert from_str(~"10") == some(10u as T);
    assert from_str(~"123456789") == some(123456789u as T);
    assert from_str(~"00100") == some(100u as T);

    assert from_str(~"") == none;
    assert from_str(~" ") == none;
    assert from_str(~"x") == none;
}

#[test]
#[ignore]
fn test_parse_buf() {
    import str::bytes;
    assert parse_buf(bytes(~"123"), 10u) == some(123u as T);
    assert parse_buf(bytes(~"1001"), 2u) == some(9u as T);
    assert parse_buf(bytes(~"123"), 8u) == some(83u as T);
    assert parse_buf(bytes(~"123"), 16u) == some(291u as T);
    assert parse_buf(bytes(~"ffff"), 16u) == some(65535u as T);
    assert parse_buf(bytes(~"z"), 36u) == some(35u as T);

    assert parse_buf(str::bytes(~"Z"), 10u) == none;
    assert parse_buf(str::bytes(~"_"), 2u) == none;
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn to_str_radix1() {
    uint::to_str(100u, 1u);
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn to_str_radix17() {
    uint::to_str(100u, 17u);
}

#[test]
fn test_times() {
    let ten = 10 as T;
    let mut accum = 0;
    for ten.times { accum += 1; }
    assert (accum == 10);
}

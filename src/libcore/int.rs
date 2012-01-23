/*
Module: int
*/

/*
Const: max_value

The maximum value of an integer
*/
// FIXME: Find another way to access the machine word size in a const expr
#[cfg(target_arch="x86")]
const max_value: int = (-1 << 31)-1;

#[cfg(target_arch="x86_64")]
const max_value: int = (-1 << 63)-1;

/*
Const: min_value

The minumum value of an integer
*/
#[cfg(target_arch="x86")]
const min_value: int = -1 << 31;

#[cfg(target_arch="x86_64")]
const min_value: int = -1 << 63;

/* Function: add */
pure fn add(x: int, y: int) -> int { ret x + y; }

/* Function: sub */
pure fn sub(x: int, y: int) -> int { ret x - y; }

/* Function: mul */
pure fn mul(x: int, y: int) -> int { ret x * y; }

/* Function: div */
pure fn div(x: int, y: int) -> int { ret x / y; }

/* Function: rem */
pure fn rem(x: int, y: int) -> int { ret x % y; }

/* Predicate: lt */
pure fn lt(x: int, y: int) -> bool { ret x < y; }

/* Predicate: le */
pure fn le(x: int, y: int) -> bool { ret x <= y; }

/* Predicate: eq */
pure fn eq(x: int, y: int) -> bool { ret x == y; }

/* Predicate: ne */
pure fn ne(x: int, y: int) -> bool { ret x != y; }

/* Predicate: ge */
pure fn ge(x: int, y: int) -> bool { ret x >= y; }

/* Predicate: gt */
pure fn gt(x: int, y: int) -> bool { ret x > y; }

/* Predicate: positive */
pure fn positive(x: int) -> bool { ret x > 0; }

/* Predicate: negative */
pure fn negative(x: int) -> bool { ret x < 0; }

/* Predicate: nonpositive */
pure fn nonpositive(x: int) -> bool { ret x <= 0; }

/* Predicate: nonnegative */
pure fn nonnegative(x: int) -> bool { ret x >= 0; }


// FIXME: Make sure this works with negative integers.
/*
Function: hash

Produce a uint suitable for use in a hash table
*/
fn hash(x: int) -> uint { ret x as uint; }

/*
Function: range

Iterate over the range [`lo`..`hi`)
*/
fn range(lo: int, hi: int, it: fn(int)) {
    let i = lo;
    while i < hi { it(i); i += 1; }
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
fn parse_buf(buf: [u8], radix: uint) -> int {
    if vec::len::<u8>(buf) == 0u {
        #error("parse_buf(): buf is empty");
        fail;
    }
    let i = vec::len::<u8>(buf) - 1u;
    let start = 0u;
    let power = 1;

    if buf[0] == ('-' as u8) {
        power = -1;
        start = 1u;
    }
    let n = 0;
    while true {
        let digit = char::to_digit(buf[i] as char);
        if (digit as uint) >= radix {
            fail;
        }
        n += (digit as int) * power;
        power *= radix as int;
        if i <= start { ret n; }
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
fn from_str(s: str) -> int { parse_buf(str::bytes(s), 10u) }

/*
Function: to_str

Convert to a string in a given base
*/
fn to_str(n: int, radix: uint) -> str {
    assert (0u < radix && radix <= 16u);
    ret if n < 0 {
            "-" + uint::to_str(-n as uint, radix)
        } else { uint::to_str(n as uint, radix) };
}

/*
Function: str

Convert to a string
*/
fn str(i: int) -> str { ret to_str(i, 10u); }

/*
Function: pow

Returns `base` raised to the power of `exponent`
*/
fn pow(base: int, exponent: uint) -> int {
    if exponent == 0u { ret 1; } //Not mathemtically true if [base == 0]
    if base     == 0  { ret 0; }
    let my_pow  = exponent;
    let acc     = 1;
    let multiplier = base;
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
    assert(from_str("0") == 0);
    assert(from_str("3") == 3);
    assert(from_str("10") == 10);
    assert(from_str("123456789") == 123456789);
    assert(from_str("00100") == 100);

    assert(from_str("-1") == -1);
    assert(from_str("-3") == -3);
    assert(from_str("-10") == -10);
    assert(from_str("-123456789") == -123456789);
    assert(from_str("-00100") == -100);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_from_str_fail_1() {
    from_str(" ");
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_from_str_fail_2() {
    from_str("x");
}

#[test]
fn test_parse_buf() {
    import str::bytes;
    assert (parse_buf(bytes("123"), 10u) == 123);
    assert (parse_buf(bytes("1001"), 2u) == 9);
    assert (parse_buf(bytes("123"), 8u) == 83);
    assert (parse_buf(bytes("123"), 16u) == 291);
    assert (parse_buf(bytes("ffff"), 16u) == 65535);
    assert (parse_buf(bytes("FFFF"), 16u) == 65535);
    assert (parse_buf(bytes("z"), 36u) == 35);
    assert (parse_buf(bytes("Z"), 36u) == 35);

    assert (parse_buf(bytes("-123"), 10u) == -123);
    assert (parse_buf(bytes("-1001"), 2u) == -9);
    assert (parse_buf(bytes("-123"), 8u) == -83);
    assert (parse_buf(bytes("-123"), 16u) == -291);
    assert (parse_buf(bytes("-ffff"), 16u) == -65535);
    assert (parse_buf(bytes("-FFFF"), 16u) == -65535);
    assert (parse_buf(bytes("-z"), 36u) == -35);
    assert (parse_buf(bytes("-Z"), 36u) == -35);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_parse_buf_fail_1() {
    parse_buf(str::bytes("Z"), 35u);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_parse_buf_fail_2() {
    parse_buf(str::bytes("-9"), 2u);
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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

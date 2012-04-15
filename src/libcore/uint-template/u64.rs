type T = u64;

// Type-specific functions here. These must be reexported by the
// parent module so that they appear in core::u8 and not core::u8::u8;


// FIXME: Surely we can generalize this to apply to all uint types
#[doc = "Convert to a string in a given base"]
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

    let mut s = "";

    let mut n = n;
    while n > 0u64 { s = digit(n % r64) + s; n /= r64; }
    ret s;
}

#[doc = "Convert to a string"]
fn str(n: u64) -> str { ret to_str(n, 10u); }

#[doc = "Parse a string as an unsigned integer."]
fn from_str(buf: str, radix: u64) -> option<u64> {
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

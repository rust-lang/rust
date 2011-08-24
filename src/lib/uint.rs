

fn add(x: uint, y: uint) -> uint { ret x + y; }

fn sub(x: uint, y: uint) -> uint { ret x - y; }

fn mul(x: uint, y: uint) -> uint { ret x * y; }

fn div(x: uint, y: uint) -> uint { ret x / y; }

fn rem(x: uint, y: uint) -> uint { ret x % y; }

pred lt(x: uint, y: uint) -> bool { ret x < y; }

pred le(x: uint, y: uint) -> bool { ret x <= y; }

pred eq(x: uint, y: uint) -> bool { ret x == y; }

pred ne(x: uint, y: uint) -> bool { ret x != y; }

pred ge(x: uint, y: uint) -> bool { ret x >= y; }

pred gt(x: uint, y: uint) -> bool { ret x > y; }

fn max(x: uint, y: uint) -> uint { if x > y { ret x; } ret y; }

fn min(x: uint, y: uint) -> uint { if x > y { ret y; } ret x; }

iter range(lo: uint, hi: uint) -> uint {
    let lo_ = lo;
    while lo_ < hi { put lo_; lo_ += 1u; }
}

fn next_power_of_two(n: uint) -> uint {
    let halfbits: uint = sys::rustrt::size_of::<uint>() * 4u;
    let tmp: uint = n - 1u;
    let shift: uint = 1u;
    while shift <= halfbits { tmp |= tmp >> shift; shift <<= 1u; }
    ret tmp + 1u;
}

fn parse_buf(buf: &[u8], radix: uint) -> uint {
    if vec::len::<u8>(buf) == 0u {
        log_err "parse_buf(): buf is empty";
        fail;
    }
    let i = vec::len::<u8>(buf) - 1u;
    let power = 1u;
    let n = 0u;
    while true {
        n += (buf[i] - ('0' as u8) as uint) * power;
        power *= radix;
        if i == 0u { ret n; }
        i -= 1u;
    }
    fail;
}

fn from_str(s: &str) -> uint { parse_buf(str::bytes(s), 10u) }

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
        s += str::unsafe_from_byte(digit(n % radix) as u8);
        n /= radix;
    }
    let s1: str = "";
    let len: uint = str::byte_len(s);
    while len != 0u { len -= 1u; s1 += str::unsafe_from_byte(s[len]); }
    ret s1;
}
fn str(i: uint) -> str { ret to_str(i, 10u); }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

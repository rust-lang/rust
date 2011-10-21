fn max_value() -> int {
  ret min_value() - 1;
}

fn min_value() -> int {
  ret (-1 << (sys::size_of::<int>()  * 8u as int - 1)) as int;
}


pure fn add(x: int, y: int) -> int { ret x + y; }

pure fn sub(x: int, y: int) -> int { ret x - y; }

pure fn mul(x: int, y: int) -> int { ret x * y; }

pure fn div(x: int, y: int) -> int { ret x / y; }

pure fn rem(x: int, y: int) -> int { ret x % y; }

pure fn lt(x: int, y: int) -> bool { ret x < y; }

pure fn le(x: int, y: int) -> bool { ret x <= y; }

pure fn eq(x: int, y: int) -> bool { ret x == y; }

pure fn ne(x: int, y: int) -> bool { ret x != y; }

pure fn ge(x: int, y: int) -> bool { ret x >= y; }

pure fn gt(x: int, y: int) -> bool { ret x > y; }

pure fn positive(x: int) -> bool { ret x > 0; }

pure fn negative(x: int) -> bool { ret x < 0; }

pure fn nonpositive(x: int) -> bool { ret x <= 0; }

pure fn nonnegative(x: int) -> bool { ret x >= 0; }


// FIXME: Make sure this works with negative integers.
fn hash(x: int) -> uint { ret x as uint; }

fn eq_alias(x: int, y: int) -> bool { ret x == y; }

fn range(lo: int, hi: int, it: block(int)) {
    while lo < hi { it(lo); lo += 1; }
}

fn parse_buf(buf: [u8], radix: uint) -> int {
    if vec::len::<u8>(buf) == 0u {
        log_err "parse_buf(): buf is empty";
        fail;
    }
    let i = vec::len::<u8>(buf) - 1u;
    let power = 1;
    if buf[0] == ('-' as u8) {
        power = -1;
        i -= 1u;
    }
    let n = 0;
    while true {
        n += (buf[i] - ('0' as u8) as int) * power;
        power *= radix as int;
        if i == 0u { ret n; }
        i -= 1u;
    }
    fail;
}

fn from_str(s: str) -> int { parse_buf(str::bytes(s), 10u) }

fn to_str(n: int, radix: uint) -> str {
    assert (0u < radix && radix <= 16u);
    ret if n < 0 {
            "-" + uint::to_str(-n as uint, radix)
        } else { uint::to_str(n as uint, radix) };
}
fn str(i: int) -> str { ret to_str(i, 10u); }

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
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

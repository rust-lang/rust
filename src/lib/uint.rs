
fn add(uint x, uint y) -> uint { ret x + y; }
fn sub(uint x, uint y) -> uint { ret x - y; }
fn mul(uint x, uint y) -> uint { ret x * y; }
fn div(uint x, uint y) -> uint { ret x / y; }
fn rem(uint x, uint y) -> uint { ret x % y; }

fn lt(uint x, uint y) -> bool { ret x < y; }
fn le(uint x, uint y) -> bool { ret x <= y; }
fn eq(uint x, uint y) -> bool { ret x == y; }
fn ne(uint x, uint y) -> bool { ret x != y; }
fn ge(uint x, uint y) -> bool { ret x >= y; }
fn gt(uint x, uint y) -> bool { ret x > y; }

fn max(uint x, uint y) -> uint {
    if (x > y) { ret x; }
    ret y;
}

iter range(uint lo, uint hi) -> uint {
    auto lo_ = lo;
    while (lo_ < hi) {
        put lo_;
        lo_ += 1u;
    }
}

fn next_power_of_two(uint n) -> uint {
    // FIXME change |* uint(4)| below to |* uint(8) / uint(2)| and watch the
    // world explode.
    let uint halfbits = sys::rustrt::size_of[uint]() * 4u;
    let uint tmp = n - 1u;
    let uint shift = 1u;
    while (shift <= halfbits) {
        tmp |= tmp >> shift;
        shift <<= 1u;
    }
    ret tmp + 1u;
}

fn parse_buf(vec[u8] buf, uint radix) -> uint {
    if (vec::len[u8](buf) == 0u) {
        log_err "parse_buf(): buf is empty";
        fail;
    }

    auto i = vec::len[u8](buf) - 1u;
    auto power = 1u;
    auto n = 0u;
    while (true) {
        n += (((buf.(i)) - ('0' as u8)) as uint) * power;
        power *= radix;
        if (i == 0u) { ret n; }
        i -= 1u;
    }

    fail;
}

fn to_str(uint num, uint radix) -> str
{
    auto n = num;

    assert (0u < radix && radix <= 16u);
    fn digit(uint n) -> char {
        ret alt (n) {
            case (0u) { '0' }
            case (1u) { '1' }
            case (2u) { '2' }
            case (3u) { '3' }
            case (4u) { '4' }
            case (5u) { '5' }
            case (6u) { '6' }
            case (7u) { '7' }
            case (8u) { '8' }
            case (9u) { '9' }
            case (10u) { 'a' }
            case (11u) { 'b' }
            case (12u) { 'c' }
            case (13u) { 'd' }
            case (14u) { 'e' }
            case (15u) { 'f' }
            case (_) { fail }
        };
    }

    if (n == 0u) { ret "0"; }

    let str s = "";
    while (n != 0u) {
        s += str::unsafe_from_byte(digit(n % radix) as u8);
        n /= radix;
    }

    let str s1 = "";
    let uint len = str::byte_len(s);
    while (len != 0u) {
        len -= 1u;
        s1 += str::unsafe_from_byte(s.(len));
    }
    ret s1;

}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

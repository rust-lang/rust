

fn add(int x, int y) -> int { ret x + y; }

fn sub(int x, int y) -> int { ret x - y; }

fn mul(int x, int y) -> int { ret x * y; }

fn div(int x, int y) -> int { ret x / y; }

fn rem(int x, int y) -> int { ret x % y; }

fn lt(int x, int y) -> bool { ret x < y; }

fn le(int x, int y) -> bool { ret x <= y; }

fn eq(int x, int y) -> bool { ret x == y; }

fn ne(int x, int y) -> bool { ret x != y; }

fn ge(int x, int y) -> bool { ret x >= y; }

fn gt(int x, int y) -> bool { ret x > y; }

fn positive(int x) -> bool { ret x > 0; }

fn negative(int x) -> bool { ret x < 0; }

fn nonpositive(int x) -> bool { ret x <= 0; }

fn nonnegative(int x) -> bool { ret x >= 0; }


// FIXME: Make sure this works with negative integers.
fn hash(&int x) -> uint { ret x as uint; }

fn eq_alias(&int x, &int y) -> bool { ret x == y; }

iter range(int lo, int hi) -> int {
    let int lo_ = lo;
    while (lo_ < hi) { put lo_; lo_ += 1; }
}

fn to_str(int n, uint radix) -> str {
    assert (0u < radix && radix <= 16u);
    ret if (n < 0) {
            "-" + uint::to_str(-n as uint, radix)
        } else { uint::to_str(n as uint, radix) };
}
fn str(int i) -> str { ret to_str(i, 10u); }

fn pow(int base, uint exponent) -> int {
    ret if (exponent == 0u) {
            1
        } else if (base == 0) {
            0
        } else {
            auto accum = base;
            auto count = exponent;
            while (count > 1u) { accum *= base; count -= 1u; }
            accum
        };
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

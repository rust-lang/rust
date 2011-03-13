
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

iter range(mutable int lo, int hi) -> int {
    while (lo < hi) {
        put lo;
        lo += 1;
    }
}

fn to_str(mutable int n, uint radix) -> str
{
    check (0u < radix && radix <= 16u);
    if (n < 0) {
        ret "-" + _uint.to_str((-n) as uint, radix);
    } else {
        ret _uint.to_str(n as uint, radix);
    }
}

fn pow(int base, uint exponent) -> int {

    if (exponent == 0u) {
        ret 1;
    } else if (base == 0) {
        ret 0;
    } else {
        auto accum = base;
        auto count = exponent;
        while (count > 1u) {
            accum *= base;
            count -= 1u;
        }
        ret accum;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

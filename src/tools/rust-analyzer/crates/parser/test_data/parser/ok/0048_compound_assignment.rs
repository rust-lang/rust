// https://github.com/rust-lang/rust-analyzer/pull/983

fn compound_assignment() {
    let mut a = 0;
    a += 1;
    a -= 2;
    a *= 3;
    a %= 4;
    a /= 5;
    a |= 6;
    a &= 7;
    a ^= 8;
    a <= 9;
    a >= 10;
    a >>= 11;
    a <<= 12;
}

fn binding_power() {
    let x = 1 + 2 * 3 % 4 - 5 / 6;
    1 + 2 * 3;
    1 << 2 + 3;
    1 & 2 >> 3;
    1 ^ 2 & 3;
    1 | 2 ^ 3;
    1 == 2 | 3;
    1 && 2 == 3;
    //1 || 2 && 2;
    //1 .. 2 || 3;
    //1 = 2 .. 3;
    //---&*1 - --2 * 9;
}

fn right_associative() {
    a = b = c;
    a = b += c -= d;
    a = b *= c /= d %= e;
    a = b &= c |= d ^= e;
    a = b <<= c >>= d;
}

fn mixed_associativity() {
    // (a + b) = (c += ((d * e) = f))
    a + b = c += d * e = f;
}

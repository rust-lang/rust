// -*- rust -*-

#[cfg(target_arch = "x86")]
fn target() {
    assert (-1000 >> 3 == 536870787);
}

#[cfg(target_arch = "x86_64")]
fn target() {
    assert (-1000 >> 3 == 2305843009213693827);
}

fn general() {
    let mut a: int = 1;
    let mut b: int = 2;
    a ^= b;
    b ^= a;
    a = a ^ b;
    log(debug, a);
    log(debug, b);
    assert (b == 1);
    assert (a == 2);
    assert (!0xf0 & 0xff == 0xf);
    assert (0xf0 | 0xf == 0xff);
    assert (0xf << 4 == 0xf0);
    assert (0xf0 >> 4 == 0xf);
    assert (-16 >>> 2 == -4);
    assert (0b1010_1010 | 0b0101_0101 == 0xff);
}

fn main() {
    general();
    target();
}

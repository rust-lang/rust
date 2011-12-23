

fn main() {
    let i32_a: int = 10;
    assert (i32_a == 10);
    assert (i32_a - 10 == 0);
    assert (i32_a / 10 == 1);
    assert (i32_a - 20 == -10);
    assert (i32_a << 10 == 10240);
    assert (i32_a << 16 == 655360);
    assert (i32_a * 16 == 160);
    assert (i32_a * i32_a * i32_a == 1000);
    assert (i32_a * i32_a * i32_a * i32_a == 10000);
    assert (i32_a * i32_a / i32_a * i32_a == 100);
    assert (i32_a * (i32_a - 1) << 2 + i32_a == 368640);
    let i32_b: int = 0x10101010;
    assert (i32_b + 1 - 1 == i32_b);
    assert (i32_b << 1 == i32_b << 1);
    assert (i32_b >> 1 == i32_b >> 1);
    assert (i32_b & i32_b << 1 == 0);
    log(debug, i32_b | i32_b << 1);
    assert (i32_b | i32_b << 1 == 0x30303030);
}

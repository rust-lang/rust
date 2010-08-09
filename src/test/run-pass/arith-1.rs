fn main() -> () {
    let int i32_a = 10;
    check(i32_a == 10);
    check(i32_a - 10 == 0);
    check(i32_a / 10 == 1);
    check(i32_a - 20 == -10);
    check(i32_a << 10 == 10240);
    check(i32_a << 16 == 655360);
    check(i32_a * 16 == 160);
    check(i32_a * i32_a * i32_a == 1000);
    check(i32_a * i32_a * i32_a * i32_a == 10000);
    check(((i32_a * i32_a) / i32_a) * i32_a == 100);
    check(i32_a * (i32_a - 1) << 2 + i32_a == 368640);

    let int i32_b = 0x10101010;
    check(i32_b + 1 - 1 == i32_b);
    check(i32_b << 1 == i32_b << 1);
    check(i32_b >> 1 == i32_b >> 1);
    check((i32_b & (i32_b << 1)) == 0);
    log ((i32_b | (i32_b << 1)));
    check((i32_b | (i32_b << 1)) == 0x30303030);
}
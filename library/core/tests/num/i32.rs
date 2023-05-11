int_module!(i32);

#[test]
fn test_arith_operation() {
    let a: isize = 10;
    assert_eq!(a * (a - 1), 90);
    let i32_a: isize = 10;
    assert_eq!(i32_a, 10);
    assert_eq!(i32_a - 10, 0);
    assert_eq!(i32_a / 10, 1);
    assert_eq!(i32_a - 20, -10);
    assert_eq!(i32_a << 10, 10240);
    assert_eq!(i32_a << 16, 655360);
    assert_eq!(i32_a * 16, 160);
    assert_eq!(i32_a * i32_a * i32_a, 1000);
    assert_eq!(i32_a * i32_a * i32_a * i32_a, 10000);
    assert_eq!(i32_a * i32_a / i32_a * i32_a, 100);
    assert_eq!(i32_a * (i32_a - 1) << (2 + i32_a as usize), 368640);
    let i32_b: isize = 0x10101010;
    assert_eq!(i32_b + 1 - 1, i32_b);
    assert_eq!(i32_b << 1, i32_b << 1);
    assert_eq!(i32_b >> 1, i32_b >> 1);
    assert_eq!(i32_b & i32_b << 1, 0);
    assert_eq!(i32_b | i32_b << 1, 0x30303030);
    let i32_c: isize = 0x10101010;
    assert_eq!(
        i32_c + i32_c * 2 / 3 * 2 + (i32_c - 7 % 3),
        i32_c + i32_c * 2 / 3 * 2 + (i32_c - 7 % 3)
    );
}

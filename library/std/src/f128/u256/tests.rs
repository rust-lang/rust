use crate::f128::u256::U256;

#[test]
fn test_shl_u128() {
    let a = 0x0123456789abcdef0123456789abcdef;

    let b = U256::shl_u128(a, 0);
    assert_eq!(b.high, 0);
    assert_eq!(b.low, a);

    let b = U256::shl_u128(a, 40);
    assert_eq!(b.high, 0x0123456789);
    assert_eq!(b.low, 0xabcdef0123456789abcdef0000000000);
}

#[test]
fn test_leading_zeros() {
    assert_eq!(U256 { high: 0, low: 0 }.leading_zeros(), 256);
    assert_eq!(U256 { high: 0xfff, low: 0xf }.leading_zeros(), 116);
    assert_eq!(U256 { high: 0, low: 0xf }.leading_zeros(), 252);
}

#[test]
fn test_sub() {
    let a = U256 { high: 7, low: 10 };
    let b = U256 { high: 3, low: 2 };
    let c = U256 { high: 3, low: 15 };
    assert_eq!(a - b, U256 { high: 4, low: 8 });
    assert_eq!(a - c, U256 { high: 3, low: u128::MAX - 4 });
}

#[test]
fn test_div_rem() {
    let a = U256 { high: 0, low: 7 };
    let b = U256 { high: 0xabcdef, low: 0x0123456789abcdef0123456789abcdef };
    let c = 0x9876543210abc;
    assert_eq!(a.div_rem(c), (0, 7));
    assert_eq!(b.div_rem(c), (0x1207a42fc2c725f9cf302ff31d, 0x7c11e593922a3));
}

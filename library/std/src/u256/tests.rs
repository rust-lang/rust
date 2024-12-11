use crate::u256::U256;

#[test]
fn test_leading_zeros() {
    assert_eq!(U256 { high: 0, low: 0 }.leading_zeros(), 256);
    assert_eq!(U256 { high: 0xfff, low: 0xf }.leading_zeros(), 116);
    assert_eq!(U256 { high: 0, low: 0xf }.leading_zeros(), 252);
}

#[test]
fn test_wrap_u128() {
    assert_eq!(U256 { high: 3, low: 4 }.wrap_u128(), 4);
}

#[test]
fn test_from_u128() {
    assert_eq!(U256::from(7u128), U256 { high: 0, low: 7 });
}

#[test]
fn test_shl() {
    let a = U256 { high: 0x1234, low: 0x0123456789abcdef0123456789abcdef };

    assert_eq!(a << 0, a);
    assert_eq!(a << 16, U256 { high: 0x12340123, low: 0x456789abcdef0123456789abcdef0000 });
    assert_eq!(a << 128, U256 { high: 0x0123456789abcdef0123456789abcdef, low: 0 });
    assert_eq!(a << 140, U256 { high: 0x3456789abcdef0123456789abcdef000, low: 0 });
    assert_eq!(a << 256, U256::ZERO);
    assert_eq!(a << 1000, U256::ZERO);
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
    let a = U256 { high: 0xabcdef, low: 0x0123456789abcdef0123456789abcdef };
    let b = U256 { high: 0, low: 0x987 };
    let c = U256 { high: 0x13, low: 0x9876543210abc };
    assert_eq!(
        a.div_rem(b),
        (U256 { high: 0x1208, low: 0x63d18447855b62f52e9a6983a9e22813 }, U256 {
            high: 0,
            low: 0xea
        })
    );
    assert_eq!(
        a.div_rem(c),
        (U256 { high: 0, low: 0x90ad6 }, U256 {
            high: 0xd,
            low: 0x0123456789abcd98d752c5f8c1057cc7
        },)
    );
    assert_eq!(c.div_rem(a), (U256::ZERO, c));
}

#[test]
fn test_div() {
    assert_eq!(U256::from(47u128) / U256::from(10u128), U256::from(4u128));
}

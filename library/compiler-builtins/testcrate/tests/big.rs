use compiler_builtins::int::{i256, u256, HInt, MinInt};

const LOHI_SPLIT: u128 = 0xaaaaaaaaaaaaaaaaffffffffffffffff;

/// Print a `u256` as hex since we can't add format implementations
fn hexu(v: u256) -> String {
    format!(
        "0x{:016x}{:016x}{:016x}{:016x}",
        v.0[3], v.0[2], v.0[1], v.0[0]
    )
}

#[test]
fn widen_u128() {
    assert_eq!(u128::MAX.widen(), u256([u64::MAX, u64::MAX, 0, 0]));
    assert_eq!(
        LOHI_SPLIT.widen(),
        u256([u64::MAX, 0xaaaaaaaaaaaaaaaa, 0, 0])
    );
}

#[test]
fn widen_i128() {
    assert_eq!((-1i128).widen(), u256::MAX.signed());
    assert_eq!(
        (LOHI_SPLIT as i128).widen(),
        i256([u64::MAX, 0xaaaaaaaaaaaaaaaa, u64::MAX, u64::MAX])
    );
    assert_eq!((-1i128).zero_widen().unsigned(), (u128::MAX).widen());
}

#[test]
fn widen_mul_u128() {
    let tests = [
        (u128::MAX / 2, 2_u128, u256([u64::MAX - 1, u64::MAX, 0, 0])),
        (u128::MAX, 2_u128, u256([u64::MAX - 1, u64::MAX, 1, 0])),
        (u128::MAX, u128::MAX, u256([1, 0, u64::MAX - 1, u64::MAX])),
        (u128::MIN, u128::MIN, u256::ZERO),
        (1234, 0, u256::ZERO),
        (0, 1234, u256::ZERO),
    ];

    let mut errors = Vec::new();
    for (i, (a, b, exp)) in tests.iter().copied().enumerate() {
        let res = a.widen_mul(b);
        let res_z = a.zero_widen_mul(b);
        assert_eq!(res, res_z);
        if res != exp {
            errors.push((i, a, b, exp, res));
        }
    }

    for (i, a, b, exp, res) in &errors {
        eprintln!(
            "FAILURE ({i}): {a:#034x} * {b:#034x} = {} got {}",
            hexu(*exp),
            hexu(*res)
        );
    }
    assert!(errors.is_empty());
}

use super::mul_div_u64;

#[test]
fn test_muldiv() {
    assert_eq!(mul_div_u64(1_000_000_000_001, 1_000_000_000, 1_000_000), 1_000_000_000_001_000);
}

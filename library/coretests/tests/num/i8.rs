int_module!(i8, u8);

#[test]
fn test_signed_div_rem_exhaustive() {
    // For i8 an exhaustive test only involves ~65k operations, and since the
    // implementations are generic, any bugs are likely to exist in all
    // variants. So an exhaustive check here is useful for all types.
    for a in i8::MIN..=i8::MAX {
        for b in i8::MIN..=i8::MAX {
            if b == 0 || a == i8::MIN && b == -1 {
                assert!(std::panic::catch_unwind(|| a.div_euclid(b)).is_err());
                assert!(std::panic::catch_unwind(|| a.rem_euclid(b)).is_err());
                assert!(std::panic::catch_unwind(|| a.div_floor(b)).is_err());
                assert!(std::panic::catch_unwind(|| a.rem_floor(b)).is_err());
                assert!(std::panic::catch_unwind(|| a.div_ceil(b)).is_err());
                assert!(std::panic::catch_unwind(|| a.rem_ceil(b)).is_err());
            } else {
                let r_euclid = a.rem_euclid(b);
                assert!(r_euclid.unsigned_abs() < b.unsigned_abs() && r_euclid >= 0);
                assert_eq!(a.div_euclid(b) * b + r_euclid, a);

                let r_floor = a.rem_floor(b);
                assert!(
                    r_floor.unsigned_abs() < b.unsigned_abs()
                        && (r_floor == 0 || (r_floor < 0) == (b < 0))
                );
                assert_eq!(a.div_floor(b) * b + r_floor, a);

                let r_ceil = a.rem_ceil(b);
                assert!(
                    r_ceil.unsigned_abs() < b.unsigned_abs()
                        && (r_floor == 0 || (r_ceil < 0) != (b < 0))
                );
                assert_eq!(a.div_ceil(b) * b + r_ceil, a);
            }
        }
    }
}

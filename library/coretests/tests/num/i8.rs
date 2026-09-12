int_module!(i8, u8);

#[test]
fn test_signed_div_rem_exhaustive() {
    // For i8 an exhaustive test only involves ~65k operations, and since the
    // implementations are generic, any bugs are likely to exist in all
    // variants. So an exhaustive check here is useful for all types.
    for a in i8::MIN..=i8::MAX {
        for b in i8::MIN..=i8::MAX {
            if b == 0 || a == i8::MIN && b == -1 {
                continue; // Tested in int_macros.rs.
            } else {
                let valid_q_r = |a, b, q, r| q as i32 * b as i32 + r as i32 == a as i32;
                let r_euclid = a.rem_euclid(b);
                assert!(r_euclid.unsigned_abs() < b.unsigned_abs() && r_euclid >= 0);
                assert!(valid_q_r(a, b, a.div_euclid(b), r_euclid));

                let r_floor = a.rem_floor(b);
                assert!(
                    r_floor.unsigned_abs() < b.unsigned_abs()
                        && (r_floor == 0 || (r_floor < 0) == (b < 0))
                );
                assert!(valid_q_r(a, b, a.div_floor(b), r_floor));

                let r_ceil = a.rem_ceil(b);
                assert!(
                    r_ceil.unsigned_abs() < b.unsigned_abs()
                        && (r_ceil == 0 || (r_ceil < 0) != (b < 0))
                );
                assert!(valid_q_r(a, b, a.div_ceil(b), r_ceil));
            }
        }
    }
}

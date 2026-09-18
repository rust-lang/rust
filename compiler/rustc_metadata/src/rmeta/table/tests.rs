use super::MinBytesToEncode;

fn min_bytes_to_encode_reference_impl(bytes: &[u8]) -> usize {
    bytes.len() - bytes.iter().rev().take_while(|&&b| b == 0).count()
}

#[test]
fn min_bytes_to_encode_1() {
    // Exhaustively test every 1-byte value.
    for i in 0..=u8::MAX {
        let bytes = [i];
        assert_eq!(bytes.min_bytes_to_encode(), min_bytes_to_encode_reference_impl(&bytes));
    }
}

#[test]
fn min_bytes_to_encode_8() {
    // Test an assortment of 8-byte values.
    let mut cases = Vec::with_capacity(512);
    for i in 0..64 {
        let base = 1u64 << i;
        cases.push(base.wrapping_sub(1));
        cases.push(base);
        cases.push(base.wrapping_add(1));
        cases.push(base.wrapping_add(0xFFFFFFFF));
        cases.push((!base).wrapping_sub(1));
        cases.push(!base);
        cases.push((!base).wrapping_add(1));
        cases.push(base & !0xFF);
        cases.push(base & !0xFF00);
    }
    cases.sort();
    cases.dedup();

    for case in cases {
        let bytes = case.to_le_bytes();
        assert_eq!(bytes.min_bytes_to_encode(), min_bytes_to_encode_reference_impl(&bytes));
    }
}

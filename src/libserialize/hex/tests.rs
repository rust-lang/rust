extern crate test;
use test::Bencher;
use crate::hex::{FromHex, ToHex};

#[test]
pub fn test_to_hex() {
    assert_eq!("foobar".as_bytes().to_hex(), "666f6f626172");
}

#[test]
pub fn test_from_hex_okay() {
    assert_eq!("666f6f626172".from_hex().unwrap(),
               b"foobar");
    assert_eq!("666F6F626172".from_hex().unwrap(),
               b"foobar");
}

#[test]
pub fn test_from_hex_odd_len() {
    assert!("666".from_hex().is_err());
    assert!("66 6".from_hex().is_err());
}

#[test]
pub fn test_from_hex_invalid_char() {
    assert!("66y6".from_hex().is_err());
}

#[test]
pub fn test_from_hex_ignores_whitespace() {
    assert_eq!("666f 6f6\r\n26172 ".from_hex().unwrap(),
               b"foobar");
}

#[test]
pub fn test_to_hex_all_bytes() {
    for i in 0..256 {
        assert_eq!([i as u8].to_hex(), format!("{:02x}", i as usize));
    }
}

#[test]
pub fn test_from_hex_all_bytes() {
    for i in 0..256 {
        let ii: &[u8] = &[i as u8];
        assert_eq!(format!("{:02x}", i as usize).from_hex()
                                               .unwrap(),
                   ii);
        assert_eq!(format!("{:02X}", i as usize).from_hex()
                                               .unwrap(),
                   ii);
    }
}

#[bench]
pub fn bench_to_hex(b: &mut Bencher) {
    let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
             ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
    b.iter(|| {
        s.as_bytes().to_hex();
    });
    b.bytes = s.len() as u64;
}

#[bench]
pub fn bench_from_hex(b: &mut Bencher) {
    let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
             ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
    let sb = s.as_bytes().to_hex();
    b.iter(|| {
        sb.from_hex().unwrap();
    });
    b.bytes = sb.len() as u64;
}

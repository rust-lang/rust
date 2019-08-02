extern crate serialize as rustc_serialize;
use rustc_serialize::leb128::*;

macro_rules! impl_test_unsigned_leb128 {
    ($test_name:ident, $write_fn_name:ident, $read_fn_name:ident, $int_ty:ident) => (
        #[test]
        fn $test_name() {
            let mut stream = Vec::new();

            for x in 0..62 {
                $write_fn_name(&mut stream, (3u64 << x) as $int_ty);
            }

            let mut position = 0;
            for x in 0..62 {
                let expected = (3u64 << x) as $int_ty;
                let (actual, bytes_read) = $read_fn_name(&stream[position ..]);
                assert_eq!(expected, actual);
                position += bytes_read;
            }
            assert_eq!(stream.len(), position);
        }
    )
}

impl_test_unsigned_leb128!(test_u16_leb128, write_u16_leb128, read_u16_leb128, u16);
impl_test_unsigned_leb128!(test_u32_leb128, write_u32_leb128, read_u32_leb128, u32);
impl_test_unsigned_leb128!(test_u64_leb128, write_u64_leb128, read_u64_leb128, u64);
impl_test_unsigned_leb128!(test_u128_leb128, write_u128_leb128, read_u128_leb128, u128);
impl_test_unsigned_leb128!(test_usize_leb128, write_usize_leb128, read_usize_leb128, usize);

#[test]
fn test_signed_leb128() {
    let values: Vec<_> = (-500..500).map(|i| i * 0x12345789ABCDEF).collect();
    let mut stream = Vec::new();
    for &x in &values {
        write_signed_leb128(&mut stream, x);
    }
    let mut pos = 0;
    for &x in &values {
        let (value, bytes_read) = read_signed_leb128(&mut stream, pos);
        pos += bytes_read;
        assert_eq!(x, value);
    }
    assert_eq!(pos, stream.len());
}

/* FIXME(#110395)
#[test]
fn from() {
    use core::convert::TryFrom;
    use core::num::TryFromIntError;

    // From
    const FROM: i64 = i64::from(1i32);
    assert_eq!(FROM, 1i64);

    // From int to float
    const FROM_F64: f64 = f64::from(42u8);
    assert_eq!(FROM_F64, 42f64);

    // Upper bounded
    const U8_FROM_U16: Result<u8, TryFromIntError> = u8::try_from(1u16);
    assert_eq!(U8_FROM_U16, Ok(1u8));

    // Both bounded
    const I8_FROM_I16: Result<i8, TryFromIntError> = i8::try_from(1i16);
    assert_eq!(I8_FROM_I16, Ok(1i8));

    // Lower bounded
    const I16_FROM_U16: Result<i16, TryFromIntError> = i16::try_from(1u16);
    assert_eq!(I16_FROM_U16, Ok(1i16));
}
*/

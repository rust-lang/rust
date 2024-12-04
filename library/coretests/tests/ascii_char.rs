use core::ascii::Char;
use core::fmt::Write;

/// Tests Display implementation for ascii::Char.
#[test]
fn test_display() {
    let want = (0..128u8).map(|b| b as char).collect::<String>();
    let mut got = String::with_capacity(128);
    for byte in 0..128 {
        write!(&mut got, "{}", Char::from_u8(byte).unwrap()).unwrap();
    }
    assert_eq!(want, got);
}

/// Tests Debug implementation for ascii::Char.
#[test]
fn test_debug_control() {
    for byte in 0..128u8 {
        let mut want = format!("{:?}", byte as char);
        // `char` uses `'\u{#}'` representation where ascii::char uses `'\x##'`.
        // Transform former into the latter.
        if let Some(rest) = want.strip_prefix("'\\u{") {
            want = format!("'\\x{:0>2}'", rest.strip_suffix("}'").unwrap());
        }
        let chr = core::ascii::Char::from_u8(byte).unwrap();
        assert_eq!(want, format!("{chr:?}"), "byte: {byte}");
    }
}

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
fn test_debug() {
    for (chr, want) in [
        // Control characters
        (Char::Null, r#"'\0'"#),
        (Char::StartOfHeading, r#"'\x01'"#),
        (Char::StartOfText, r#"'\x02'"#),
        (Char::EndOfText, r#"'\x03'"#),
        (Char::EndOfTransmission, r#"'\x04'"#),
        (Char::Enquiry, r#"'\x05'"#),
        (Char::Acknowledge, r#"'\x06'"#),
        (Char::Bell, r#"'\x07'"#),
        (Char::Backspace, r#"'\x08'"#),
        (Char::CharacterTabulation, r#"'\t'"#),
        (Char::LineFeed, r#"'\n'"#),
        (Char::LineTabulation, r#"'\x0b'"#),
        (Char::FormFeed, r#"'\x0c'"#),
        (Char::CarriageReturn, r#"'\r'"#),
        (Char::ShiftOut, r#"'\x0e'"#),
        (Char::ShiftIn, r#"'\x0f'"#),
        (Char::DataLinkEscape, r#"'\x10'"#),
        (Char::DeviceControlOne, r#"'\x11'"#),
        (Char::DeviceControlTwo, r#"'\x12'"#),
        (Char::DeviceControlThree, r#"'\x13'"#),
        (Char::DeviceControlFour, r#"'\x14'"#),
        (Char::NegativeAcknowledge, r#"'\x15'"#),
        (Char::SynchronousIdle, r#"'\x16'"#),
        (Char::EndOfTransmissionBlock, r#"'\x17'"#),
        (Char::Cancel, r#"'\x18'"#),
        (Char::EndOfMedium, r#"'\x19'"#),
        (Char::Substitute, r#"'\x1a'"#),
        (Char::Escape, r#"'\x1b'"#),
        (Char::InformationSeparatorFour, r#"'\x1c'"#),
        (Char::InformationSeparatorThree, r#"'\x1d'"#),
        (Char::InformationSeparatorTwo, r#"'\x1e'"#),
        (Char::InformationSeparatorOne, r#"'\x1f'"#),
        // U+007F is also a control character
        (Char::Delete, r#"'\x7f'"#),

        // Characters which need escaping.
        (Char::ReverseSolidus, r#"'\\'"#),
        (Char::Apostrophe, r#"'\''"#),

        // Regular, non-control characters, which don’t need any special
        // handling.
        (Char::Space, r#"' '"#),
        (Char::QuotationMark, r#"'"'"#),
        (Char::CapitalM, r#"'M'"#),
        (Char::Tilde, r#"'~'"#),
    ] {
        assert_eq!(want, format!("{chr:?}"), "chr: {}", chr as u8);
    }
}

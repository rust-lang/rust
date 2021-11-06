pub const TEXT_FLOW_CONTROL_CHARS: &[char] = &[
    '\u{202A}', '\u{202B}', '\u{202D}', '\u{202E}', '\u{2066}', '\u{2067}', '\u{2068}', '\u{202C}',
    '\u{2069}',
];

#[inline]
pub fn contains_text_flow_control_chars(s: &str) -> bool {
    // Char   - UTF-8
    // U+202A - E2 80 AA
    // U+202B - E2 80 AB
    // U+202C - E2 80 AC
    // U+202D - E2 80 AD
    // U+202E - E2 80 AE
    // U+2066 - E2 81 A6
    // U+2067 - E2 81 A7
    // U+2068 - E2 81 A8
    // U+2069 - E2 81 A9
    let mut bytes = s.as_bytes();
    loop {
        match core::slice::memchr::memchr(0xE2, &bytes) {
            Some(idx) => {
                // bytes are valid UTF-8 -> E2 must be followed by two bytes
                let ch = &bytes[idx..idx + 3];
                match ch {
                    [_, 0x80, 0xAA..=0xAE] | [_, 0x81, 0xA6..=0xA9] => break true,
                    _ => {}
                }
                bytes = &bytes[idx + 3..];
            }
            None => {
                break false;
            }
        }
    }
}

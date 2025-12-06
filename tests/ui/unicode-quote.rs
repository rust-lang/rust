// Provide a useful error message when attempting to escape a non-ascii quotation mark.
// <https://github.com/rust-lang/rust/issues/128858>
fn main() {
    dbg!("\″"); //U+2033
    //~^ERROR unknown character escape: `\u{2033}`
    dbg!("\ʺ"); //U+02BA
    //~^ERROR 6:12: 6:13: unknown character escape: `\u{2ba}`
    dbg!("\˝"); //U+02DD
    //~^ERROR 8:12: 8:13: unknown character escape: `\u{2dd}`
    dbg!("\̋"); //U+030B
    //~^ERROR 10:12: 10:13: unknown character escape: `\u{30b}`
    dbg!("\̎"); //U+030E
    //~^ERROR 12:12: 12:13: unknown character escape: `\u{30e}`
    dbg!("\״"); //U+05F4
    //~^ERROR 14:12: 14:13: unknown character escape: `\u{5f4}`
    dbg!("\“"); //U+201C
    //~^ERROR 16:12: 16:13: unknown character escape: `\u{201c}`
    dbg!("\”"); //U+201D
    //~^ERROR 18:12: 18:13: unknown character escape: `\u{201d}`
}

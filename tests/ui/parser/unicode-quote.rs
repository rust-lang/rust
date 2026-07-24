// Provide a useful error message when attempting to escape a non-ASCII quotation mark.
// <https://github.com/rust-lang/rust/issues/128858>
fn main() {
    dbg!("\″"); //U+2033
    //~^ ERROR unknown character escape: `\u{2033}`
    dbg!("\ʺ"); //U+02BA
    //~^ ERROR unknown character escape: `\u{2ba}`
    dbg!("\˝"); //U+02DD
    //~^ ERROR unknown character escape: `\u{2dd}`
    dbg!("\̋"); //U+030B
    //~^ ERROR unknown character escape: `\u{30b}`
    dbg!("\̎"); //U+030E
    //~^ ERROR unknown character escape: `\u{30e}`
    dbg!("\״"); //U+05F4
    //~^ ERROR unknown character escape: `\u{5f4}`
    dbg!("\“"); //U+201C
    //~^ ERROR unknown character escape: `\u{201c}`
    dbg!("\”"); //U+201D
    //~^ ERROR unknown character escape: `\u{201d}`
}

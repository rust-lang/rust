const FOO: Option<&[[u8; 3]]> = Some(&[*b"foo"]); //~ ERROR temporary value dropped while borrowed

use std::borrow::Cow;

pub const X: [u8; 3] = *b"ABC";
pub const Y: Cow<'static, [ [u8; 3] ]> = Cow::Borrowed(&[X]);


pub const Z: Cow<'static, [ [u8; 3] ]> = Cow::Borrowed(&[*b"ABC"]);
//~^ ERROR temporary value dropped while borrowed

fn main() {}

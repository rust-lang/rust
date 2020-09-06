use super::*;
use crate::fmt::{Debug, Formatter, Result};

#[test]
fn smoke() {
    struct Helper<'a>(&'a [u8]);

    impl Debug for Helper<'_> {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            debug_fmt_bytestring(self.0, f)
        }
    }

    let input = b"\xF0hello,\tworld";
    let expected = r#""\xF0hello,\tworld""#;
    let output = format!("{:?}", Helper(input));

    assert!(output == expected);
}

//@run-rustfix
#![allow(clippy::needless_raw_string_hashes, clippy::no_effect, unused)]
#![warn(clippy::needless_raw_strings)]
#![feature(c_str_literals)]

fn main() {
    r#"aaa"#;
    r#""aaa""#;
    r#"\s"#;
    br#"aaa"#;
    br#""aaa""#;
    br#"\s"#;
    // currently disabled: https://github.com/rust-lang/rust/issues/113333
    // cr#"aaa"#;
    // cr#""aaa""#;
    // cr#"\s"#;
}

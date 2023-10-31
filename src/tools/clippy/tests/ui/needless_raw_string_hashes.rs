//@run-rustfix
#![allow(clippy::no_effect, unused)]
#![warn(clippy::needless_raw_string_hashes)]
#![feature(c_str_literals)]

fn main() {
    r#"aaa"#;
    r##"Hello "world"!"##;
    r######" "### "## "# "######;
    r######" "aa" "# "## "######;
    br#"aaa"#;
    br##"Hello "world"!"##;
    br######" "### "## "# "######;
    br######" "aa" "# "## "######;
    // currently disabled: https://github.com/rust-lang/rust/issues/113333
    // cr#"aaa"#;
    // cr##"Hello "world"!"##;
    // cr######" "### "## "# "######;
    // cr######" "aa" "# "## "######;
}

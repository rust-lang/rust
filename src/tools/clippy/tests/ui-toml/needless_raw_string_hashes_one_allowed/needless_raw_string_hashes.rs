#![allow(clippy::no_effect, unused)]
#![warn(clippy::needless_raw_string_hashes)]

fn main() {
    r#"\aaa"#;
    r##"\aaa"##;
    //~^ needless_raw_string_hashes
    r##"Hello "world"!"##;
    //~^ needless_raw_string_hashes
    r######" "### "## "# "######;
    //~^ needless_raw_string_hashes
}

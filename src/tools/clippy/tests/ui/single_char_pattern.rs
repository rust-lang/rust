#![allow(clippy::needless_raw_strings, clippy::needless_raw_string_hashes, unused_must_use)]
#![warn(clippy::single_char_pattern)]
use std::collections::HashSet;

fn main() {
    let x = "foo";
    x.split("x");
    //~^ single_char_pattern
    x.split("xx");
    x.split('x');

    let y = "x";
    x.split(y);
    x.split("ß");
    x.split("ℝ");
    x.split("💣");
    // Can't use this lint for unicode code points which don't fit in a char
    x.split("❤️");
    x.split_inclusive("x");
    //~^ single_char_pattern
    x.contains("x");
    //~^ single_char_pattern
    x.starts_with("x");
    //~^ single_char_pattern
    x.ends_with("x");
    //~^ single_char_pattern
    x.find("x");
    //~^ single_char_pattern
    x.rfind("x");
    //~^ single_char_pattern
    x.rsplit("x");
    //~^ single_char_pattern
    x.split_terminator("x");
    //~^ single_char_pattern
    x.rsplit_terminator("x");
    //~^ single_char_pattern
    x.splitn(2, "x");
    //~^ single_char_pattern
    x.rsplitn(2, "x");
    //~^ single_char_pattern
    x.split_once("x");
    //~^ single_char_pattern
    x.rsplit_once("x");
    //~^ single_char_pattern
    x.matches("x");
    //~^ single_char_pattern
    x.rmatches("x");
    //~^ single_char_pattern
    x.match_indices("x");
    //~^ single_char_pattern
    x.rmatch_indices("x");
    //~^ single_char_pattern
    x.trim_start_matches("x");
    //~^ single_char_pattern
    x.trim_end_matches("x");
    //~^ single_char_pattern
    x.replace("x", "y");
    //~^ single_char_pattern
    x.replacen("x", "y", 3);
    //~^ single_char_pattern
    // Make sure we escape characters correctly.
    x.split("\n");
    //~^ single_char_pattern
    x.split("'");
    //~^ single_char_pattern
    x.split("\'");
    //~^ single_char_pattern
    // Issue #11973: Don't escape `"` in `'"'`
    x.split("\"");
    //~^ single_char_pattern

    let h = HashSet::<String>::new();
    h.contains("X"); // should not warn

    x.replace(';', ",").split(","); // issue #2978
    //
    //~^^ single_char_pattern
    x.starts_with("\x03"); // issue #2996
    //
    //~^^ single_char_pattern

    // Issue #3204
    const S: &str = "#";
    x.find(S);

    // Raw string
    x.split(r"a");
    //~^ single_char_pattern
    x.split(r#"a"#);
    //~^ single_char_pattern
    x.split(r###"a"###);
    //~^ single_char_pattern
    x.split(r###"'"###);
    //~^ single_char_pattern
    x.split(r###"#"###);
    //~^ single_char_pattern
    // Must escape backslash in raw strings when converting to char #8060
    x.split(r#"\"#);
    //~^ single_char_pattern
    x.split(r"\");
    //~^ single_char_pattern

    // should not warn, the char versions are actually slower in some cases
    x.strip_prefix("x");
    x.strip_suffix("x");
}

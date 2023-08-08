//@run-rustfix

#![allow(clippy::needless_raw_strings, clippy::needless_raw_string_hashes, unused_must_use)]

use std::collections::HashSet;

fn main() {
    let x = "foo";
    x.split("x");
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
    x.contains("x");
    x.starts_with("x");
    x.ends_with("x");
    x.find("x");
    x.rfind("x");
    x.rsplit("x");
    x.split_terminator("x");
    x.rsplit_terminator("x");
    x.splitn(2, "x");
    x.rsplitn(2, "x");
    x.split_once("x");
    x.rsplit_once("x");
    x.matches("x");
    x.rmatches("x");
    x.match_indices("x");
    x.rmatch_indices("x");
    x.trim_start_matches("x");
    x.trim_end_matches("x");
    x.strip_prefix("x");
    x.strip_suffix("x");
    x.replace("x", "y");
    x.replacen("x", "y", 3);
    // Make sure we escape characters correctly.
    x.split("\n");
    x.split("'");
    x.split("\'");

    let h = HashSet::<String>::new();
    h.contains("X"); // should not warn

    x.replace(';', ",").split(","); // issue #2978
    x.starts_with("\x03"); // issue #2996

    // Issue #3204
    const S: &str = "#";
    x.find(S);

    // Raw string
    x.split(r"a");
    x.split(r#"a"#);
    x.split(r###"a"###);
    x.split(r###"'"###);
    x.split(r###"#"###);
    // Must escape backslash in raw strings when converting to char #8060
    x.split(r#"\"#);
    x.split(r"\");
}

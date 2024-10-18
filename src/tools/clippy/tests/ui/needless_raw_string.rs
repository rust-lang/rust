#![allow(clippy::needless_raw_string_hashes, clippy::no_effect, unused)]
#![warn(clippy::needless_raw_strings)]

fn main() {
    r#"aaa"#;
    r#""aaa""#;
    r#"\s"#;
    br#"aaa"#;
    br#""aaa""#;
    br#"\s"#;
    cr#"aaa"#;
    cr#""aaa""#;
    cr#"\s"#;

    r#"
        a
        multiline
        string
    "#;

    r"no hashes";
    br"no hashes";
    cr"no hashes";
}

fn issue_13503() {
    println!(r"SELECT * FROM posts");
    println!(r#"SELECT * FROM posts"#);
    println!(r##"SELECT * FROM "posts""##);

    // Test arguments as well
    println!("{}", r"foobar".len());
}

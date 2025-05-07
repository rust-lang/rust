#![allow(clippy::needless_raw_string_hashes, clippy::no_effect, unused)]
#![warn(clippy::needless_raw_strings)]

fn main() {
    r#"aaa"#;
    //~^ needless_raw_strings
    r#""aaa""#;
    r#"\s"#;
    br#"aaa"#;
    //~^ needless_raw_strings
    br#""aaa""#;
    br#"\s"#;
    cr#"aaa"#;
    //~^ needless_raw_strings
    cr#""aaa""#;
    cr#"\s"#;

    r#"
    //~^ needless_raw_strings
        a
        multiline
        string
    "#;

    r"no hashes";
    //~^ needless_raw_strings
    br"no hashes";
    //~^ needless_raw_strings
    cr"no hashes";
    //~^ needless_raw_strings
}

fn issue_13503() {
    println!(r"SELECT * FROM posts");
    //~^ needless_raw_strings
    println!(r#"SELECT * FROM posts"#);
    //~^ needless_raw_strings
    println!(r##"SELECT * FROM "posts""##);

    // Test arguments as well
    println!("{}", r"foobar".len());
    //~^ needless_raw_strings
}

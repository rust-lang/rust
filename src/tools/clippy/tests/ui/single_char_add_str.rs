#![warn(clippy::single_char_add_str)]
#![allow(clippy::needless_raw_strings, clippy::needless_raw_string_hashes)]

macro_rules! get_string {
    () => {
        String::from("Hello world!")
    };
}

fn main() {
    // `push_str` tests

    let mut string = String::new();
    string.push_str("R");
    string.push_str("'");

    string.push('u');
    string.push_str("st");
    string.push_str("");
    string.push_str("\x52");
    string.push_str("\u{0052}");
    string.push_str(r##"a"##);

    let c_ref = &'a';
    string.push_str(&c_ref.to_string());
    let c = 'a';
    string.push_str(&c.to_string());
    string.push_str(&'a'.to_string());

    get_string!().push_str("ö");

    // `insert_str` tests

    let mut string = String::new();
    string.insert_str(0, "R");
    string.insert_str(1, "'");

    string.insert(0, 'u');
    string.insert_str(2, "st");
    string.insert_str(0, "");
    string.insert_str(0, "\x52");
    string.insert_str(0, "\u{0052}");
    let x: usize = 2;
    string.insert_str(x, r##"a"##);
    const Y: usize = 1;
    string.insert_str(Y, r##"a"##);
    string.insert_str(Y, r##"""##);
    string.insert_str(Y, r##"'"##);

    string.insert_str(0, &c_ref.to_string());
    string.insert_str(0, &c.to_string());
    string.insert_str(0, &'a'.to_string());

    get_string!().insert_str(1, "?");
}

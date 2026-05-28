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
    //~^ single_char_add_str
    string.push_str("'");
    //~^ single_char_add_str

    string.push('u');
    string.push_str("st");
    string.push_str("");
    string.push_str("\x52");
    //~^ single_char_add_str
    string.push_str("\u{0052}");
    //~^ single_char_add_str
    string.push_str(r##"a"##);
    //~^ single_char_add_str

    let c_ref = &'a';
    string.push_str(&c_ref.to_string());
    //~^ single_char_add_str
    let c = 'a';
    string.push_str(&c.to_string());
    //~^ single_char_add_str
    string.push_str(&'a'.to_string());
    //~^ single_char_add_str

    get_string!().push_str("ö");
    //~^ single_char_add_str

    // `insert_str` tests

    let mut string = String::new();
    string.insert_str(0, "R");
    //~^ single_char_add_str
    string.insert_str(1, "'");
    //~^ single_char_add_str

    string.insert(0, 'u');
    string.insert_str(2, "st");
    string.insert_str(0, "");
    string.insert_str(0, "\x52");
    //~^ single_char_add_str
    string.insert_str(0, "\u{0052}");
    //~^ single_char_add_str
    let x: usize = 2;
    string.insert_str(x, r##"a"##);
    //~^ single_char_add_str
    const Y: usize = 1;
    string.insert_str(Y, r##"a"##);
    //~^ single_char_add_str
    string.insert_str(Y, r##"""##);
    //~^ single_char_add_str
    string.insert_str(Y, r##"'"##);
    //~^ single_char_add_str

    string.insert_str(0, &c_ref.to_string());
    //~^ single_char_add_str
    string.insert_str(0, &c.to_string());
    //~^ single_char_add_str
    string.insert_str(0, &'a'.to_string());
    //~^ single_char_add_str

    get_string!().insert_str(1, "?");
    //~^ single_char_add_str
}

// run-rustfix
#![warn(clippy::single_char_push_str)]

macro_rules! get_string {
    () => {
        String::from("Hello world!")
    };
}

fn main() {
    let mut string = String::new();
    string.push_str("R");
    string.push_str("'");

    string.push('u');
    string.push_str("st");
    string.push_str("");
    string.push_str("\x52");
    string.push_str("\u{0052}");
    string.push_str(r##"a"##);

    get_string!().push_str("ö");
}

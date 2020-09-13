// run-rustfix
#![warn(clippy::single_char_push_str)]

fn main() {
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
}

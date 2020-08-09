// run-rustfix
#![warn(clippy::single_char_push_str)]

fn main() {
    let mut string = String::new();
    string.push_str("R");
    string.push_str("'");

    string.push('u');
}

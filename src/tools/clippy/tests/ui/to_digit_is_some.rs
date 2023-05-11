//@run-rustfix

#![warn(clippy::to_digit_is_some)]

fn main() {
    let c = 'x';
    let d = &c;

    let _ = d.to_digit(8).is_some();
    let _ = char::to_digit(c, 8).is_some();
}

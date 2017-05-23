#![feature(plugin)]
#![plugin(clippy)]

#[warn(zero_width_space)]
fn zero() {
    print!("Here >​< is a ZWS, and ​another");
    print!("This\u{200B}is\u{200B}fine");
}

#[warn(unicode_not_nfc)]
fn canon() {
    print!("̀àh?");
    print!("a\u{0300}h?"); // also okay
}

#[warn(non_ascii_literal)]
fn uni() {
    print!("Üben!");
    print!("\u{DC}ben!"); // this is okay
}

fn main() {
    zero();
    uni();
    canon();
}

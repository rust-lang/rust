#![feature(plugin)]
#![plugin(clippy)]

#[deny(zero_width_space)]
fn zero() {
    print!("Here >​< is a ZWS, and ​another");
               //~^ ERROR zero-width space detected
    print!("This\u{200B}is\u{200B}fine");
}

#[deny(unicode_not_nfc)]
fn canon() {
    print!("̀àh?"); //~ERROR non-nfc unicode sequence detected
    print!("a\u{0300}h?"); // also okay
}

#[deny(non_ascii_literal)]
fn uni() {
    print!("Üben!"); //~ERROR literal non-ASCII character detected
    print!("\u{DC}ben!"); // this is okay
}

fn main() {
    zero();
    uni();
    canon();
}

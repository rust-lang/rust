#![feature(plugin)]
#![plugin(clippy)]

#[deny(zero_width_space)]
fn zero() {
    print!("Here >​< is a ZWS, and ​another");
               //~^ ERROR zero-width space detected
}

#[deny(unicode_not_nfc)]
fn canon() {
    print!("̀àh?"); //~ERROR non-nfc unicode sequence detected
}

#[deny(non_ascii_literal)]
fn uni() {
    print!("Üben!"); //~ERROR literal non-ASCII character detected
}

fn main() {
    zero();
    uni();
    canon();
}

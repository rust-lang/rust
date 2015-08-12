#![feature(plugin)]
#![plugin(clippy)]

#[deny(zero_width_space)]
fn zero() {
    print!("Here >​< is a ZWS, and ​another");
               //~^ ERROR zero-width space detected. Consider using `\u{200B}`
                            //~^^ ERROR zero-width space detected. Consider using `\u{200B}`
}

//#[deny(unicode_canon)]
fn canon() {
    print!("̀ah?"); //not yet ~ERROR non-canonical unicode sequence detected. Consider using à
}

#[deny(non_ascii_literal)]
fn uni() {
    print!("Üben!"); //~ERROR literal non-ASCII character detected. Consider using `\u{DC}`
}

fn main() {
    zero();
    uni();
    canon();
}

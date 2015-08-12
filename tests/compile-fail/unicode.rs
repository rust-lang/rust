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

//#[deny(ascii_only)]
fn uni() {
    println!("Üben!"); //not yet ~ERROR Unicode literal detected. Consider using \u{FC}
}

fn main() {
    zero();
    uni();
    canon();
}

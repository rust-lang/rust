//@ run-rustfix
#[allow(unused_parens)]
fn main() {
    let _ = +1; //~ ERROR leading `+` is not supported
    let _ = (1.0 + +2.0) * +3.0; //~ ERROR leading `+` is not supported
                           //~| ERROR leading `+` is not supported
    let _ = [+3, 4+6]; //~ ERROR leading `+` is not supported
}

// run-pass
// ignore-pretty `dyn ::foo` parses differently in the current edition

use std::fmt::Display;

static BYTE: u8 = 33;

fn main() {
    let x: &(dyn 'static + Display) = &BYTE;
    let y: Box<dyn Display + 'static> = Box::new(BYTE);
    let _: &dyn (Display) = &BYTE;
    let _: &dyn (::std::fmt::Display) = &BYTE;
    let xstr = format!("{}", x);
    let ystr = format!("{}", y);
    assert_eq!(xstr, "33");
    assert_eq!(ystr, "33");
}

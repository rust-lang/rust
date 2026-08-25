//@ revisions: e2015 e2021

//@[e2021] edition: 2021
//@[e2015] edition: 2015
//@[e2015] check-pass

// Make sure we reject the case where a raw lifetime is immediately followed by another
// lifetime. This reserves a modest amount of space for changing lexing to, for example,
// delay rejection of overlong char literals like `'r#long'id`.

macro_rules! w {
    ($($tt:tt)*) => {}
}

w!('r#long'id);
//[e2021]~^ ERROR character literal may only contain one codepoint

fn main() {}

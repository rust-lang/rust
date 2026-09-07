// Tests for https://github.com/rust-lang/rust/issues/101487
// These C-style reference cases should not
// provide machine-applicable suggestions

macro_rules! m {
    ($e:expr) => { $e };
    ($($t:tt)*) => { 0 };
}

fn a() {
    let _c: u8 & u8;
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `=`, found `&`
}

fn b() {
    let _c: u8 &mut;
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `=`, found `&`
}

fn main() {
    let x = 12;
    let y = 34;

    let a = x + y&;
    //~^ ERROR reference types must be written as `&expr`

    let _d = x as u8&;
    //~^ ERROR reference types must be written as `&expr`

    m!(x &);
    //~^ ERROR expected expression, found end of macro arguments
}

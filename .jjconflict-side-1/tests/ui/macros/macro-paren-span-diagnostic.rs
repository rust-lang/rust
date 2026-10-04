//! Check that error spans in parenthesized macro expressions point to the call site.

#[rustfmt::skip]
macro_rules! paren {
    ($e:expr) => (($e))
    //            ^^^^ do not highlight here
}

mod m {
    pub struct S {
        x: i32,
    }

    pub fn make() -> S {
        S { x: 0 }
    }
}

fn main() {
    let s = m::make();
    paren!(s.x); //~ ERROR field `x` of struct `S` is private
    //     ^^^ highlight here
}

// Be smart about span of parenthesized expression in macro.

macro_rules! paren {
    ($e:expr) => (($e))
    //            ^^^^ do not highlight here
}

mod m {
    pub struct S {
        x: i32
    }
    pub fn make() -> S {
        S { x: 0 }
    }
}

fn main() {
    let s = m::make();
    paren!(s.x); //~ ERROR field `x` of struct `m::S` is private
    //     ^^^ highlight here
}

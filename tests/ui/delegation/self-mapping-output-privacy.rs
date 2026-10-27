#![feature(fn_delegation)]

trait Trait {
    fn method(&self) -> Self;
}

pub struct S;
impl Trait for S {
    fn method(&self) -> S {
        S
    }
}

mod private {
    pub struct W(super::S);

    pub struct W1 {
        sss: super::S,
    }
}

impl Trait for private::W {
    reuse Trait::method { S }
    //~^ ERROR: field `0` of struct `W` is private
}

impl private::W {
    reuse Trait::method { S }
    //~^ ERROR: field `0` of struct `W` is private
}

impl Trait for private::W1 {
    reuse Trait::method { S }
    //~^ ERROR: field `sss` of struct `W1` is private
}

impl private::W1 {
    reuse Trait::method { S }
    //~^ ERROR: field `sss` of struct `W1` is private
}

fn main() {}

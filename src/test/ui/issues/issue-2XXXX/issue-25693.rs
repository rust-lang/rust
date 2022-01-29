// run-pass
#![allow(unused_variables)]
pub trait Parameters { type SelfRef; }

struct RP<'a> { _marker: std::marker::PhantomData<&'a ()> }
struct BP;

impl<'a> Parameters for RP<'a> { type SelfRef = &'a X<RP<'a>>; }
impl Parameters for BP { type SelfRef = Box<X<BP>>; }

pub struct Y;
pub enum X<P: Parameters> {
    Nothing,
    SameAgain(P::SelfRef, Y)
}

fn main() {
    let bnil: Box<X<BP>> = Box::new(X::Nothing);
    let bx: Box<X<BP>> = Box::new(X::SameAgain(bnil, Y));
    let rnil: X<RP> = X::Nothing;
    let rx: X<RP> = X::SameAgain(&rnil, Y);
}

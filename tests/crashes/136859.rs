//@ known-bug: #136859
#![feature(generic_const_exprs)]

trait If<const COND: bool> {}
impl If<true> for () {}

trait IsZero<const N: u8> {
    type Answer;
}

struct True;
struct False;

impl<const N: u8> IsZero<N> for ()
where (): If<{N == 0}> {
    type Msg = True;
}

trait Foobar<const N: u8> {}

impl<const N: u8> Foobar<N> for ()
where (): IsZero<N, Answer = True> {}

impl<const N: u8> Foobar<{{ N }}> for ()
where (): IsZero<N, Answer = False> {}

fn main() {}

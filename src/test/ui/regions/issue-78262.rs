// revisions: default nll polonius
// ignore-compare-mode-nll
// ignore-compare-mode-polonius
// [nll] compile-flags: -Z borrowck=mir
// [polonius] compile-flags: -Z borrowck=mir -Z polonius

trait TT {}

impl dyn TT {
    fn func(&self) {}
}

fn main() {
    let f = |x: &dyn TT| x.func(); //[default]~ ERROR: mismatched types
    //[nll]~^ ERROR: borrowed data escapes outside of closure
    //[polonius]~^^ ERROR: borrowed data escapes outside of closure
}

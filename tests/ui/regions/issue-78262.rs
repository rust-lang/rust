//@ revisions: base polonius
//@ ignore-compare-mode-polonius
//@ [polonius] compile-flags: -Z polonius

trait TT {}

impl dyn TT {
    fn func(&self) {}
}

fn main() {
    let f = |x: &dyn TT| x.func();
    //[base]~^ ERROR: borrowed data escapes outside of closure
    //[polonius]~^^ ERROR: borrowed data escapes outside of closure
}

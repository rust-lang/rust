// revisions: nll default
// ignore-compare-mode-nll
//[nll]compile-flags: -Z borrowck=mir

trait TT {}

impl dyn TT {
    fn func(&self) {}
}

fn main() {
    let f = |x: &dyn TT| x.func(); //[default]~ ERROR: mismatched types
    //[nll]~^ ERROR: borrowed data escapes outside of closure
}

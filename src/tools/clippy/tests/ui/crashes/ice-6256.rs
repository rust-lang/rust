// originally from rustc ./tests/ui/regions/issue-78262.rs
// ICE: to get the signature of a closure, use args.as_closure().sig() not fn_sig()
#![allow(clippy::upper_case_acronyms)]

trait TT {}

impl dyn TT {
    fn func(&self) {}
}

#[rustfmt::skip]
fn main() {
    let f = |x: &dyn TT| x.func();
    //~^ ERROR: borrowed data escapes outside of closure
}

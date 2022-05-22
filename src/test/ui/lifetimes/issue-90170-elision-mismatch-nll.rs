// FIXME(nll): On NLL stabilization, this should be replace
// `issue-90170-elision-mismatch.rs`. Compiletest has
// problems with rustfix and revisions.
// ignore-compare-mode-nll
// compile-flags: -Zborrowck=mir

// run-rustfix

pub fn foo(x: &mut Vec<&u8>, y: &u8) { x.push(y); } //~ ERROR lifetime may not live long enough

pub fn foo2(x: &mut Vec<&'_ u8>, y: &u8) { x.push(y); } //~ ERROR lifetime may not live long enough

pub fn foo3<'a>(_other: &'a [u8], x: &mut Vec<&u8>, y: &u8) { x.push(y); } //~ ERROR lifetime may not live long enough

fn main() {}

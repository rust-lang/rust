// build-pass (FIXME(62277): could be check-pass?)
// Tests for an LLVM abort when storing a lifetime-parametric fn into
// context that is expecting one that is not lifetime-parametric
// (i.e., has no `for <'_>`).

pub struct A<'a>(&'a ());
pub struct S<T>(T);

pub fn bad<'s>(v: &mut S<fn(A<'s>)>, y: S<for<'b> fn(A<'b>)>) {
    *v = y;
}

fn main() {}

// Check that inliner handles various forms of recursion and doesn't fall into
// an infinite inlining cycle. The particular outcome of inlining is not
// crucial otherwise.
//
// Regression test for issue #78573.

fn main() {
    one();
    two();
}

// EMIT_MIR inline_cycle.one.Inline.diff
fn one() {
    <C as Call>::call();
}

pub trait Call {
    fn call();
}

pub struct A<T>(T);
pub struct B<T>(T);
pub struct C;

impl<T: Call> Call for A<T> {
    #[inline]
    fn call() {
        <B<T> as Call>::call()
    }
}


impl<T: Call> Call for B<T> {
    #[inline]
    fn call() {
        <T as Call>::call()
    }
}

impl Call for C {
    #[inline]
    fn call() {
        A::<C>::call()
    }
}

// EMIT_MIR inline_cycle.two.Inline.diff
fn two() {
    call(f);
}

#[inline]
fn call<F: FnOnce()>(f: F) {
    f();
}

#[inline]
fn f() {
    call(f);
}

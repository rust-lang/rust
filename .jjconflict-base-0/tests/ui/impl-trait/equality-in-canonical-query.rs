// issue: #116877
//@ revisions: sized clone
//@ check-pass

#[cfg(sized)] fn rpit() -> impl Sized {}
#[cfg(clone)] fn rpit() -> impl Clone {}

fn same_output<Out>(_: impl Fn() -> Out, _: impl Fn() -> Out) {}

pub fn foo() -> impl Sized {
    same_output(rpit, foo);
    same_output(foo, rpit);
    rpit()
}

fn main () {}

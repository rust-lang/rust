//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-fail
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// Make sure we can't trick the compiler by using a projection.

trait Cat<'a> {}
impl Cat<'_> for () {}

trait Id {
    type Id;
}
impl<T> Id for T {
    type Id = T;
}

struct S<T> {
    tail: <T as Id>::Id,
}

fn m<'a>() {
    let unsend: *const dyn Cat<'a> = &();
    let _send = unsend as *const S<dyn Cat<'static>>;
    //~^ error: lifetime may not live long enough
}

fn main() {
    m();
}

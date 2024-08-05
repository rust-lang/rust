//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-fail
//
// Make sure we can't trick the compiler by using a projection.

trait Cat<'a> {}
impl Cat<'_> for () {}

trait Id {
    type Id: ?Sized;
}
impl<T: ?Sized> Id for T {
    type Id = T;
}

struct S<T: ?Sized> {
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

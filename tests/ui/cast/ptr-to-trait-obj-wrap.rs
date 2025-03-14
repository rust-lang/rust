// Checks that various casts of pointers to trait objects wrapped in structures
// work. Note that the metadata doesn't change when a DST is wrapped in a
// structure, so these casts *are* fine.
//
//@ check-pass

trait A {}

struct W<T: ?Sized>(T);
struct X<T: ?Sized>(T);

fn unwrap(a: *const W<dyn A>) -> *const dyn A {
    a as _
}

fn unwrap_nested(a: *const W<W<dyn A>>) -> *const W<dyn A> {
    a as _
}

fn rewrap(a: *const W<dyn A>) -> *const X<dyn A> {
    a as _
}

fn rewrap_nested(a: *const W<W<dyn A>>) -> *const W<X<dyn A>> {
    a as _
}

fn wrap(a: *const dyn A) -> *const W<dyn A> {
    a as _
}

fn main() {}

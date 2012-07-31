// A dummy trait/impl that work close over any type.  The trait will
// be parameterized by a region due to the &self/int constraint.

trait foo {
    fn foo(i: &self/int) -> int;
}

impl<T:copy> of foo for T {
    fn foo(i: &self/int) -> int {*i}
}

fn to_foo<T:copy>(t: T) {
    // This version is ok because, although T may contain borrowed
    // pointers, it never escapes the fn body.  We know this because
    // the type of foo includes a region which will be resolved to
    // the fn body itself.
    let v = &3;
    let x = {f:t} as foo;
    assert x.foo(v) == 3;
}

fn to_foo_2<T:copy>(t: T) -> foo {
    // Not OK---T may contain borrowed ptrs and it is going to escape
    // as part of the returned foo value
    {f:t} as foo //~ ERROR value may contain borrowed pointers; use `owned` bound
}

fn to_foo_3<T:copy owned>(t: T) -> foo {
    // OK---T may escape as part of the returned foo value, but it is
    // owned and hence does not contain borrowed ptrs
    {f:t} as foo
}

fn main() {
}
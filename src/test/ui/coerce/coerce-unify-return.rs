// run-pass
// Check that coercions unify the expected return type of a polymorphic
// function call, instead of leaving the type variables as they were.

// pretty-expanded FIXME #23616

struct Foo;
impl Foo {
    fn foo<T>(self, x: T) -> Option<T> { Some(x) }
}

pub fn main() {
    let _: Option<fn()> = Some(main);
    let _: Option<fn()> = Foo.foo(main);

    // The same two cases, with implicit type variables made explicit.
    let _: Option<fn()> = Some::<_>(main);
    let _: Option<fn()> = Foo.foo::<_>(main);
}

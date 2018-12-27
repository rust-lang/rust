// Test that attempts to construct infinite types via impl trait fail
// in a graceful way.
//
// Regression test for #38064.

// error-pattern:overflow evaluating the requirement `impl Quux`

trait Quux {}

fn foo() -> impl Quux {
    struct Foo<T>(T);
    impl<T> Quux for Foo<T> {}
    Foo(bar())
}

fn bar() -> impl Quux {
    struct Bar<T>(T);
    impl<T> Quux for Bar<T> {}
    Bar(foo())
}

// effectively:
//     struct Foo(Bar);
//     struct Bar(Foo);
// should produce an error about infinite size

fn main() { foo(); }

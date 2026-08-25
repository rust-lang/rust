//@ known-bug: #154964
//@ edition: 2021
//@ compile-flags: -Zprint-type-sizes
fn main() {
    |foo: Foo<_>| async { foo.bar };
}
struct Foo<F> {
    bar: (),
}

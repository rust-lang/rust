//@ known-bug: #117942
struct Foo {
    _: union  {
    #[rustfmt::skip]
    f: String
    },
}

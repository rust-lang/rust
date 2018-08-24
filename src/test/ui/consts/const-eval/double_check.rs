// compile-pass

enum Foo {
    A = 5,
    B = 42,
}
enum Bar {
    C = 42,
    D = 99,
}
union Union {
    foo: &'static Foo,
    bar: &'static Bar,
    usize: &'static usize,
}
static BAR: usize = 42;
static FOO: (&Foo, &Bar) = unsafe {(
    Union { usize: &BAR }.foo,
    Union { usize: &BAR }.bar,
)};

fn main() {}

//@ check-pass

struct Foo {
    foo: Option<&'static Foo>
}

static FOO: Foo = Foo {
    foo: Some(&FOO),
};

fn main() {}

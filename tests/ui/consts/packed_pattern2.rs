//@ run-pass

#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(packed)]
struct Foo {
    field: (u8, u16),
}

#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(align(2))]
struct Bar {
    a: Foo,
}

const FOO: Bar = Bar {
    a: Foo {
        field: (5, 6),
    }
};

fn main() {
    match FOO {
        Bar { a: Foo { field: (5, 6) } } => {},
        FOO => unreachable!(), //~ WARNING unreachable pattern
        _ => unreachable!(),
    }
}

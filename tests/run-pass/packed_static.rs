#[repr(packed)]
struct Foo {
    i: i32
}

fn main() {
    assert_eq!({FOO.i}, 42);
}

static FOO: Foo = Foo { i: 42 };

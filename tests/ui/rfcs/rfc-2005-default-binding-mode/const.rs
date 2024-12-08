// FIXME(tschottdorf): this test should pass.

#[derive(PartialEq, Eq)]
struct Foo {
    bar: i32,
}

const FOO: Foo = Foo{bar: 5};

fn main() {
    let f = Foo{bar:6};

    match &f {
        FOO => {}, //~ ERROR mismatched types
        _ => panic!(),
    }
}

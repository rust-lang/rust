//@ run-pass

struct Foo;

static X: Foo = Foo;

pub fn main() {
    match X {
        Foo => {}
    }
}

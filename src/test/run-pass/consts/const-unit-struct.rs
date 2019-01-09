// run-pass
// pretty-expanded FIXME #23616

struct Foo;

static X: Foo = Foo;

pub fn main() {
    match X {
        Foo => {}
    }
}

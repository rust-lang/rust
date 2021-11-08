// check-pass
// pretty-expanded FIXME #23616

#[deny(dead_code)]
pub enum Foo {
    Bar {
        baz: isize
    }
}

fn main() { }

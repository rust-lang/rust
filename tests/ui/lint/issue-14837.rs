//@ check-pass

#[deny(dead_code)]
pub enum Foo {
    Bar {
        baz: isize
    }
}

fn main() { }

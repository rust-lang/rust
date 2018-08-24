// error-pattern: cycle detected

#![feature(const_fn)]

struct Foo {
    bytes: [u8; std::mem::size_of::<Foo>()]
}

fn main() {}

// ignore-musl
// ignore-x86
// error-pattern: cycle detected

struct Foo {
    bytes: [u8; std::mem::size_of::<Foo>()]
}

fn main() {}

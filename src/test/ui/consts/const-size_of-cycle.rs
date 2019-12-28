// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
// error-pattern: cycle detected

struct Foo {
    bytes: [u8; std::mem::size_of::<Foo>()]
}

fn main() {}

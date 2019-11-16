// ignore-x86 FIXME: missing sysroot spans (#53081)
// error-pattern: cycle detected

struct Foo {
    bytes: [u8; std::mem::size_of::<Foo>()]
}

fn main() {}

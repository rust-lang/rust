//@error-in-other-file: cycle detected

struct Foo {
    bytes: [u8; std::mem::size_of::<Foo>()]
}

fn main() {}

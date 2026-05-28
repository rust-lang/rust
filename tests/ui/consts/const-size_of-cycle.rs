struct Foo {
    bytes: [u8; std::mem::size_of::<Foo>()]
    //~^ ERROR cycle detected when evaluating type-level constant
}

fn main() {}

#[repr(align(16))]
struct Foo(u64);

#[repr(align(8))] //~ ERROR `#[repr(align(x))]` on enums is experimental
enum Bar {
    Foo { foo: Foo },
    Baz,
}

fn main() { }

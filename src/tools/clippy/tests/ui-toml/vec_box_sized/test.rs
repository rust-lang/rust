struct S {
    x: u64,
}

struct C {
    y: u16,
}

struct Foo(Vec<Box<u8>>);
struct Bar(Vec<Box<u32>>);
struct Baz(Vec<Box<(u32, u32)>>);
struct BarBaz(Vec<Box<S>>);
struct FooBarBaz(Vec<Box<C>>);

fn main() {}

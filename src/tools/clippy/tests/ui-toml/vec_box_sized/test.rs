struct S {
    x: u64,
}

struct C {
    y: u16,
}

struct Foo(Vec<Box<u8>>);
//~^ vec_box
struct Bar(Vec<Box<u16>>);
//~^ vec_box
struct Quux(Vec<Box<u32>>);
struct Baz(Vec<Box<(u16, u16)>>);
struct BarBaz(Vec<Box<S>>);
struct FooBarBaz(Vec<Box<C>>);
//~^ vec_box

fn main() {}

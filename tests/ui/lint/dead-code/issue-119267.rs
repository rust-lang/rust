// run-pass

#[derive(Debug)]
struct Foo(i32);

#[derive(Debug)]
struct ConstainsDropField(Foo, #[allow(unused)] Foo);

fn main() {}

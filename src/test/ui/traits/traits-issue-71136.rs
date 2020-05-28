struct Foo(u8);

#[derive(Clone)]
struct FooHolster {
    the_foos: Vec<Foo>, //~ERROR Clone
}

fn main() {}

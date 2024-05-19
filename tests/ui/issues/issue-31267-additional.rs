//@ run-pass

#[derive(Clone, Copy, Debug)]
struct Bar;

const BAZ: Bar = Bar;

#[derive(Debug)]
struct Foo(#[allow(dead_code)] [Bar; 1]);

struct Biz;

impl Biz {
    const BAZ: Foo = Foo([BAZ; 1]);
}

fn main() {
    let foo = Biz::BAZ;
    println!("{:?}", foo);
}

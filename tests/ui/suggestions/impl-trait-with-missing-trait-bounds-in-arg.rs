//@ run-rustfix

trait Foo {}

trait Bar {
    fn hello(&self) {}
}

struct S;

impl Foo for S {}
impl Bar for S {}

fn test(foo: impl Foo) {
    foo.hello(); //~ ERROR no method named `hello` found
}

trait Trait {
    fn method(&self) {}
}

impl Trait for fn() {}

#[allow(dead_code)]
fn test2(f: impl Fn() -> dyn std::fmt::Debug) {
    f.method(); //~ ERROR no method named `method` found
}

fn main() {
    test(S);
}

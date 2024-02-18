//@ run-pass
// Test that methods whose impl-trait-ref contains associated types
// are supported.

trait Device {
    type Resources;
}
#[allow(dead_code)]
struct Foo<D, R>(D, R);

trait Tr {
    fn present(&self) {}
}

impl<D: Device> Tr for Foo<D, D::Resources> {
    fn present(&self) {}
}

struct Res;
struct Dev;
impl Device for Dev {
    type Resources = Res;
}

fn main() {
    let foo = Foo(Dev, Res);
    foo.present();
}

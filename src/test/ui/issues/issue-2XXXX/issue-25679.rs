// run-pass
trait Device {
    type Resources;
}
struct Foo<D, R>(D, R);

impl<D: Device> Foo<D, D::Resources> {
    fn present(&self) {}
}

struct Res;
struct Dev;

impl Device for Dev { type Resources = Res; }

fn main() {
    let foo = Foo(Dev, Res);
    foo.present();
}

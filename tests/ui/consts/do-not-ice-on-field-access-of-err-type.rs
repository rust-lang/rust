trait Foo {}
impl<T> Foo for T {}

fn main() {
    let array = [(); { loop {} }]; //~ ERROR constant evaluation is taking a long time

    let tup = (7,);
    let x: &dyn Foo = &tup.0;
}

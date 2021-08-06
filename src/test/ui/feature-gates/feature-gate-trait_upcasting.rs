trait Foo {}

trait Bar: Foo {}

impl Foo for () {}

impl Bar for () {}

fn main() {
    let bar: &dyn Bar = &();
    let foo: &dyn Foo = bar;
    //~^ ERROR trait upcasting coercion is experimental [E0658]
}

//@ run-rustfix

trait Foo {}

impl Foo for fn() {}

fn main() {
    let _x: &dyn Foo = &main;
    //~^ ERROR the trait bound
}

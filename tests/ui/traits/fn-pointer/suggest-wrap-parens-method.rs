//@ run-rustfix

trait Foo {}

impl Foo for fn() {}

trait Bar {
    fn do_stuff(&self) where Self: Foo {}
}
impl<T> Bar for T {}

fn main() {
    main.do_stuff();
    //~^ ERROR the trait bound
}

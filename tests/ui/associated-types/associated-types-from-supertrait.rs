//@ run-pass

trait Foo: Iterator<Item = i32> {}
trait Bar: Foo {}

fn main() {
    let _: &dyn Bar;
}

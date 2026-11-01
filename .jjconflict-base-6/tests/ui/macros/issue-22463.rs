//@ check-pass
macro_rules! items {
    () => {
        type A = ();
        fn a() {}
    }
}

trait Foo {
    type A;
    fn a();
}

impl Foo for () {
    items!();
}

fn main() {

}

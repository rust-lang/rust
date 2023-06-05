// check-pass

trait Foo {
    type Bar
    where
        Self: Sized;
}

fn foo(_: &dyn Foo) {}

fn main() {}
